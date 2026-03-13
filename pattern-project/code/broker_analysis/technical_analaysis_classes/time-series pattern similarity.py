time-series pattern similarity
The algorithm most commonly used is called Dynamic Time Warping (DTW).
DTW compares two sequences of numbers and finds how similar their shapes are, even if the timing is slightly different.


pattern you draw
        ↓
convert to numeric array
        ↓
market OHLC data
        ↓
sliding window
        ↓
DTW similarity score
        ↓
find similar patterns


Libraries:

tslearn

dtaidistance

scipy

numpy

pandas

pip install dtaidistance

from dtaidistance import dtw

pattern = [1,2,3,2,1]
data = [1,1.5,2.5,3,2.5,2,1]

distance = dtw.distance(pattern, data)
print(distance)


Even More Advanced (What Hedge Funds Do)

Instead of DTW they use:

CNN pattern recognition

Transformer time-series models

Matrix Profile motif discovery

But DTW is the best starting point