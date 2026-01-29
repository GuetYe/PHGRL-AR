x = 1:100;
y = sin(x/12) + 0.2*randn(size(x));
plot_smoothed_band(x, y, 'Window', 15, 'Band', 1.5, 'ShowRaw', true);
