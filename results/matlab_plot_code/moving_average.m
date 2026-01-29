function moving_arerage_result = moving_average(original_data,window_size)
%MOVING_AVERAGE 计算输入数据的original_data的滑动平均
n = length(original_data);
y = zeros(size(original_data));

for i = 1:n
    if i < window_size
        % 前i个数据求平均
        y(i) = mean(original_data(1:i));
    else
        % 当前位置往前window_size个数据求平均
        y(i) = mean(original_data(i-window_size+1:i));
    end
end
moving_arerage_result = y;
end

