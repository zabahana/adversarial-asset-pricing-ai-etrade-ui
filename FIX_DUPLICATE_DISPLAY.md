# Fix for Duplicate Display Issue

## Problem
The UI repeats/repeats contents after analysis completes. This is because `display_results()` is being called multiple times or the results section is being rendered multiple times due to Streamlit reruns.

## Solution
1. Add a `results_displayed` flag that is set to `True` after first display
2. Only call `display_results()` once per analysis completion
3. Reset the flag when a new analysis starts

## Changes Made
- Added `st.session_state.results_displayed = False` when analysis starts
- Check `not st.session_state.get('results_displayed', False)` before displaying
- Set `st.session_state.results_displayed = True` immediately after displaying

This ensures results are only displayed once per analysis session.

