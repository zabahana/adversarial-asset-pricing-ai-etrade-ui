# Multi-Head Attention Architecture Diagram Sketch

## Overall Layout
- **Canvas Size**: 14 units wide × 11.5 units tall
- **Background Color**: Light gray (#f5f7fa)
- **Title**: At the top center (y=10.5)

## Title
**Text**: "MHA-DQN Architecture: 8 Feature Groups → 8 Attention Heads → 3 Attention Layers"
- Position: x=7 (center), y=10.5
- Font: Monospace, Bold, Size 12
- Color: Dark blue (#1e40af)

---

## 1. INPUT FEATURE GROUPS (Top Section)
**Position**: y=9.0 (starting position)

### Layout
- **8 boxes** arranged in **2 rows of 4 boxes each**
- **Box Size**: 2.0 units wide × 1.8 units tall
- **Spacing**: 0.15 units between boxes
- **Row spacing**: 0.4 units between rows

### Box Details
Each box contains:
- **Group Number** (top): Bold white text, size 10
- **Group Name** (below number): Light blue text (#e0e7ff), size 8
- **Feature List** (vertical, bulleted): Light blue text (#c7d2fe), size 6, italic
  - Each feature on its own line with bullet point (•)
  - Show up to 6 features, then "..." if more

### Box Colors
- **Background**: Blue gradient (#2563eb to #1e40af)
- **Border**: Blue (#4299e1), 2px width

### Feature Groups (Left to Right, Top to Bottom):

**Row 1:**
1. **Group 1** - "Price & Returns"
   - • return
   - • volatility
   - • sma_20
   - • sma_50
   - • rsi_14
   - • log_return

2. **Group 2** - "Macroeconomic"
   - • inflation
   - • GDP_growth
   - • unemployment
   - • fed_funds_rate
   - • t10y2y
   - • m2_money_supply

3. **Group 3** - "Commodities"
   - • wti_crude
   - • gold
   - • silver
   - • natural_gas
   - • copper

4. **Group 4** - "Market Indices"
   - • spy
   - • qqq
   - • vix
   - • xlk
   - • xlf
   - • xle

**Row 2:**
5. **Group 5** - "Forex"
   - • eur_usd
   - • gbp_usd
   - • usd_jpy
   - • aud_usd

6. **Group 6** - "Technical"
   - • macd
   - • macd_signal
   - • bb_upper
   - • bb_lower
   - • atr
   - • adx

7. **Group 7** - "Earnings/Sentiment"
   - • earnings_reported_eps
   - • earnings_surprise_pct
   - • days_since_earnings
   - • approaching_earnings
   - • sentiment_score

8. **Group 8** - "Crypto"
   - • btc_usd
   - • eth_usd
   - • xrp_usd

---

## 2. ARROWS FROM INPUT TO HEADS
**Position**: Between input boxes (y≈7.2) and head box (y=6.8)

- **8 arrows**, one from each input box
- All arrows converge to the center of the head box (x=7)
- **Arrow Style**: 
  - Color: Gray (#64748b)
  - Width: 1.5px
  - Head width: 0.1, Head length: 0.08
  - Slightly curved or straight diagonal

---

## 3. ATTENTION HEADS (Single Condensed Box)
**Position**: y=6.8

### Box Details
- **Size**: 11 units wide × 0.6 units tall
- **Position**: x=1.5 (left edge)
- **Background**: Teal/green (#00d4aa)
- **Border**: Teal (#00d4aa), 2.5px width

### Text Content
- **Top Line** (y=6.8+0.5): "8 ATTENTION HEADS (H1-H8: Self-Attention with Q, K, V)"
  - Font: Monospace, Bold, Size 9, White
- **Bottom Line** (y=6.8+0.2): "Multi-Head Attention: Each head processes one feature group independently"
  - Font: Monospace, Size 7, Light teal (#d1fae5), Italic

---

## 4. ARROW FROM HEADS TO LAYERS
**Position**: Between head box (y=6.8) and layer box (y=5.2)

- **Single vertical arrow** from center (x=7)
- **Arrow Style**:
  - Color: Gray (#64748b)
  - Width: 2.5px
  - Head width: 0.12, Head length: 0.1
  - Perfectly vertical

---

## 5. ATTENTION LAYERS (Single Condensed Box)
**Position**: y=5.2

### Box Details
- **Size**: 11 units wide × 0.7 units tall
- **Position**: x=1.5 (left edge)
- **Background**: Purple (#7c3aed)
- **Border**: Purple (#7c3aed), 2.5px width

### Text Content
- **Top Line** (y=5.2+0.6): "3 ATTENTION LAYERS (Stacked Multi-Head Attention)"
  - Font: Monospace, Bold, Size 9, Light purple (#c4b5fd)
- **Middle Line** (y=5.2+0.45): "Layer 1 → Layer 2 → Layer 3"
  - Font: Monospace, Size 7.5, Medium purple (#a78bfa), Weight 600
- **Bottom Line** (y=5.2+0.1): "Multi-Head Attention → Layer Norm → Feed-Forward (128 → 512 → 128)"
  - Font: Monospace, Size 6.5, Light purple (#c4b5fd), Italic

---

## 6. ARROW FROM LAYERS TO POOLING
**Position**: Between layer box (y=5.2) and pooling box (y=3.4)

- **Single vertical arrow** from center (x=7)
- **Arrow Style**: Same as previous (gray, 2.5px, vertical)

---

## 7. GLOBAL AVERAGE POOLING
**Position**: y=3.4

### Box Details
- **Size**: 6 units wide × 0.5 units tall
- **Position**: x=4 (centered)
- **Background**: Purple (#7c3aed)
- **Border**: Purple (#7c3aed), 2.5px width

### Text Content
- **Top Line** (y=3.4+0.33): "Global Average Pooling"
  - Font: Monospace, Bold, Size 8.5, Light purple (#c4b5fd)
- **Bottom Line** (y=3.4+0.17): "Sequence → Single Vector (128-dim)"
  - Font: Monospace, Size 7, Medium purple (#a78bfa), Italic

---

## 8. ARROW FROM POOLING TO OUTPUT
**Position**: Between pooling box (y=3.4) and output box (y=1.4)

- **Single vertical arrow** from center (x=7)
- **Arrow Style**: Same as previous (gray, 2.5px, vertical)

---

## 9. OUTPUT: Q-VALUES
**Position**: y=1.4

### Box Details
- **Size**: 5 units wide × 0.5 units tall
- **Position**: x=4.5 (centered)
- **Background**: Orange/amber (#f59e0b)
- **Border**: Orange (#f59e0b), 2.5px width

### Text Content
- **Top Line** (y=1.4+0.45): "OUTPUT: Q-VALUES (3 Actions)"
  - Font: Monospace, Bold, Size 9, White
- **Action Labels** (y=1.4+0.08):
  - "BUY" at x=5.5
  - "HOLD" at x=7 (center)
  - "SELL" at x=8.5
  - Font: Monospace, Bold, Size 7, Light yellow (#fef3c7)

---

## Color Palette Reference
- **Input Boxes**: Blue (#2563eb to #1e40af gradient)
- **Attention Heads**: Teal/Green (#00d4aa)
- **Attention Layers**: Purple (#7c3aed)
- **Pooling**: Purple (#7c3aed)
- **Output**: Orange/Amber (#f59e0b)
- **Arrows**: Gray (#64748b)
- **Background**: Light gray (#f5f7fa)
- **Text Colors**: 
  - White for headings
  - Light blue (#e0e7ff, #c7d2fe) for feature text
  - Light teal (#d1fae5) for head descriptions
  - Light purple (#c4b5fd, #a78bfa) for layer descriptions
  - Light yellow (#fef3c7) for action labels

---

## Drawing Tips

1. **Start with the grid**: Draw a 14×11.5 unit grid lightly
2. **Draw boxes from top to bottom**: Input → Heads → Layers → Pooling → Output
3. **Add arrows last**: Connect boxes with arrows
4. **Use consistent spacing**: 0.15-0.4 units between elements
5. **Keep text readable**: Use monospace font, appropriate sizes
6. **Color coding**: Use the color palette consistently
7. **Vertical lists**: Make sure feature lists are clearly vertical with bullet points

---

## File Format
Save as **PNG** with:
- **Resolution**: 300 DPI (or higher for print quality)
- **Dimensions**: At least 1400×1150 pixels (scaled from 14×11.5 units)
- **Background**: Transparent or light gray (#f5f7fa)
- **Format**: PNG with transparency support

