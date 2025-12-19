"""
Trading logic - generate signals and manage positions
"""

import numpy as np

from config import (
    TOP_PERCENTILE, BOTTOM_PERCENTILE, MIN_CONFIDENCE,
    TP_PARAMS, SL_PARAMS, RISK_PER_TRADE, MAX_HOLD_HOURS
)


def is_signal(percentile):
    """Check if we should trade"""
    if percentile >= TOP_PERCENTILE:
        return True
    if percentile <= BOTTOM_PERCENTILE:
        return True
    return False


def get_confidence(percentile):
    """Calculate how confident we are in the signal"""
    if percentile >= TOP_PERCENTILE:
        # how far above threshold
        conf = (percentile - TOP_PERCENTILE) / (1.0 - TOP_PERCENTILE)
        return conf
    elif percentile <= BOTTOM_PERCENTILE:
        # how far below threshold
        conf = (BOTTOM_PERCENTILE - percentile) / BOTTOM_PERCENTILE
        return conf
    return 0.0


def calculate_tp_sl(direction, expected_return, confidence):
    """Calculate take profit and stop loss levels"""
    
    # get multipliers from config
    tp_mult = TP_PARAMS['base'] + confidence * TP_PARAMS['confidence']
    sl_mult = SL_PARAMS['base'] + confidence * SL_PARAMS['confidence']
    
    if direction == 'LONG':
        tp = expected_return * tp_mult
        sl = -abs(expected_return) * sl_mult
    else:
        tp = -expected_return * tp_mult
        sl = abs(expected_return) * sl_mult
    
    # make sure stop loss is not too tight
    if abs(sl) < SL_PARAMS['minimum']:
        if direction == 'LONG':
            sl = -SL_PARAMS['minimum']
        else:
            sl = SL_PARAMS['minimum']
    
    return tp, sl


def generate_signal(percentile, historical_returns, window=720):
    """
    Generate trading signal from model prediction
    
    Returns dict with trade info or None if no signal
    """
    # check if percentile is in signal range
    if not is_signal(percentile):
        return None
    
    # determine direction
    if percentile >= TOP_PERCENTILE:
        direction = 'LONG'
    else:
        direction = 'SHORT'
    
    # calculate confidence
    confidence = get_confidence(percentile)
    
    # skip low confidence signals
    if confidence < MIN_CONFIDENCE:
        return None
    
    # get expected return at this percentile
    if len(historical_returns) > window:
        recent = historical_returns[-window:]
    else:
        recent = historical_returns
    
    expected_return = np.percentile(recent, percentile * 100)
    
    # calculate tp/sl
    tp, sl = calculate_tp_sl(direction, expected_return, confidence)
    
    # risk reward ratio
    if sl != 0:
        rr = abs(tp / sl)
    else:
        rr = 0
    
    return {
        'direction': direction,
        'percentile': percentile,
        'confidence': confidence,
        'expected_return': expected_return,
        'take_profit': tp,
        'stop_loss': sl,
        'rr_ratio': rr
    }


def calculate_position_size(stop_loss):
    """Calculate how big position to take based on risk"""
    sl_distance = abs(stop_loss)
    
    if sl_distance == 0:
        return 0
    
    size = RISK_PER_TRADE / sl_distance
    return size


def check_exit(trade, current_bar, hours_open):
    """
    Check if we should close the trade
    
    Returns (should_exit, reason, pnl)
    """
    entry = trade['entry_price']
    direction = trade['direction']
    
    # check take profit and stop loss
    if direction == 'LONG':
        # TP hit - price went up enough
        if current_bar['high'] >= trade['tp_price']:
            pnl = (trade['tp_price'] - entry) / entry
            return True, 'TP', pnl
        
        # SL hit - price dropped too much
        if current_bar['low'] <= trade['sl_price']:
            pnl = (trade['sl_price'] - entry) / entry
            return True, 'SL', pnl
    
    else:  # SHORT
        # TP hit - price went down enough
        if current_bar['low'] <= trade['tp_price']:
            pnl = (entry - trade['tp_price']) / entry
            return True, 'TP', pnl
        
        # SL hit - price went up too much
        if current_bar['high'] >= trade['sl_price']:
            pnl = (entry - trade['sl_price']) / entry
            return True, 'SL', pnl
    
    # time exit - been in trade too long
    if hours_open >= MAX_HOLD_HOURS:
        current_price = current_bar['close']
        
        if direction == 'LONG':
            pnl = (current_price - entry) / entry
        else:
            pnl = (entry - current_price) / entry
        
        return True, 'TIME', pnl
    
    return False, '', 0.0