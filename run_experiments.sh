# Let's define a list of dates to train 
# and a list of dates to test
start_dates=(2021-1-01)
end_dates=(2021-6-30)
pred_start_dates=(2021-7-01)
pred_end_dates=(2021-7-31)
presets=(none sus h7 h7_sus)
# Now we can loop over the presets
for preset in ${presets[@]}; do
    # And loop over the dates
    for i in ${!start_dates[@]}; do
        start_date=${start_dates[$i]}
        end_date=${end_dates[$i]}
        pred_start_date=${pred_start_dates[$i]}
        pred_end_date=${pred_end_dates[$i]}
        # And run the experiment
        python3 train.py \
            --start_date $start_date \
            --end_date $end_date \
            --pred_start_date $pred_start_date \
            --pred_end_date $pred_end_date \
            --preset $preset
    done
done