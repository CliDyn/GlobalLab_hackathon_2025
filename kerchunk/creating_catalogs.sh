#TCO95L91-CORE2-ctl1950d

# conda activate kern
python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCO95L91-CORE2-ctl1950d/outdata/fesom \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCO95L91-CORE2-ctl1950d_ocean_native_daily.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCO95L91-CORE2-ctl1950d_ocean_native_daily/ \
                          --vars v1-31 u1-31 temp1-31 salt1-31 vice uice m_ice a_ice w1-31 \
                          --pattern "{var}.fesom.*.nc" \
                          --time-var "time"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCO95L91-CORE2-ctl1950d/outdata/fesom \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCO95L91-CORE2-ctl1950d_ocean_native_monthly.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCO95L91-CORE2-ctl1950d_ocean_native_monthly/ \
                          --vars fh MLD2 tx_sur ty_sur \
                          --pattern "{var}.fesom.*.nc" \
                          --time-var "time"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCO95L91-CORE2-ctl1950d/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCO95L91-CORE2-ctl1950d_atmos_remapped_6h.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCO95L91-CORE2-ctl1950d_atmos_remapped_6h/ \
                          --vars 10u 10v 2t hcc lcc mcc pl_w tcc \
                          --pattern "atm_remapped_6h_{var}_6h_*.nc" \
                          --time-var "time_counter"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCO95L91-CORE2-ctl1950d/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCO95L91-CORE2-ctl1950d_atmos_remapped_1d.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCO95L91-CORE2-ctl1950d_atmos_remapped_1d/ \
                          --vars cp lsp \
                          --pattern "atm_remapped_1d_{var}_1d_*.nc" \
                          --time-var "time_counter"

###########################
# TCO95L91-CORE2-ssp585d
###########################

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCO95L91-CORE2-ssp585d/outdata/fesom \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCO95L91-CORE2-ssp585d_ocean_native_daily.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCO95L91-CORE2-ssp585d_ocean_native_daily/ \
                          --vars v1-31 u1-31 temp1-31 salt1-31 vice uice m_ice a_ice w1-31 \
                          --pattern "{var}.fesom.*.nc" \
                          --time-var "time"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCO95L91-CORE2-ssp585d/outdata/fesom \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCO95L91-CORE2-ssp585d_ocean_native_monthly.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCO95L91-CORE2-ssp585d_ocean_native_monthly/ \
                          --vars fh MLD2 tx_sur ty_sur \
                          --pattern "{var}.fesom.*.nc" \
                          --time-var "time"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCO95L91-CORE2-ssp585d/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCO95L91-CORE2-ssp585d_atmos_remapped_6h.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCO95L91-CORE2-ssp585d_atmos_remapped_6h/ \
                          --vars 10u 10v 2t hcc lcc mcc pl_w tcc \
                          --pattern "atm_remapped_6h_{var}_6h_*.nc" \
                          --time-var "time_counter"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCO95L91-CORE2-ssp585d/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCO95L91-CORE2-ssp585d_atmos_remapped_1d.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCO95L91-CORE2-ctl1950d_atmos_remapped_1d/ \
                          --vars cp lsp \
                          --pattern "atm_remapped_1d_{var}_1d_*.nc" \
                          --time-var "time_counter"

###################################
# TCo319-DART-ctl1950d-gibbs-charn
##################################

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo319-DART-ctl1950d-gibbs-charn/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo319-DART-ctl1950d-gibbs-charn_remapped_6h.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo319-DART-ctl1950d-gibbs-charn_atmos_remapped_6h/ \
                          --vars 10u 10v 2t hcc lcc mcc pl_w_850 tcc \
                          --pattern "atm_remapped_6h_{var}_6h_*.nc" \
                          --time-var "time_counter"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo319-DART-ctl1950d-gibbs-charn/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo319-DART-ctl1950d-gibbs-charn_atmos_remapped_1d.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo319-DART-ctl1950d-gibbs-charn_atmos_remapped_1d/ \
                          --vars cp lsp \
                          --pattern "atm_remapped_1d_{var}_1d_*.nc" \
                          --time-var "time_counter"

###################################
# TCo319-DART-hi1950d-gibbs-charn
##################################

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo319-DART-hi1950d-gibbs-charn/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TTCo319-DART-hi1950d-gibbs-charn_remapped_1m.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo319-DART-hi1950d-gibbs-charn_atmos_remapped_1m/ \
                          --vars 10u 10v 2t cp lsp \
                          --pattern "atm_remapped_1m_{var}_1m_*.nc" \
                          --time-var "time_counter"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo319-DART-hi1950d-gibbs-charn/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TTCo319-DART-hi1950d-gibbs-charn_remapped_1d.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo319-DART-hi1950d-gibbs-charn_atmos_remapped_1d/ \
                          --vars ivt \
                          --pattern "Re_{var}_Global_AWI-CM3_hist_*.nc" \
                          --time-var "time"

###################################
# TCo319-DART-ssp585d-gibbs-charn
##################################

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo319-DART-ssp585d-gibbs-charn/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo319-DART-ssp585d-gibbs-charn_remapped_1m.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo319-DART-ssp585d-gibbs-charn_atmos_remapped_1m/ \
                          --vars 10u 10v 2t cp lsp  pl_z_200 pl_z_500 pl_z_850 \
                          --pattern "atm_remapped_1m_{var}_1m_*.nc" \
                          --time-var "time_counter"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo319-DART-ssp585d-gibbs-charn/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo319-DART-ssp585d-gibbs-charn_remapped_6h.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo319-DART-ssp585d-gibbs-charn_atmos_remapped_6h/ \
                          --vars 10u 10v 2t hcc lcc mcc pl_w_850 tcc \
                          --pattern "atm_remapped_6h_{var}_6h_*.nc" \
                          --time-var "time_counter"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo319-DART-ssp585d-gibbs-charn/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo319-DART-ssp585d-gibbs-charn_atmos_remapped_1d.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo319-DART-ssp585d-gibbs-charn_atmos_remapped_1d/ \
                          --vars cp lsp \
                          --pattern "atm_remapped_1d_{var}_1d_*.nc" \
                          --time-var "time_counter"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo319-DART-ssp585d-gibbs-charn/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo319-DART-ssp585d-gibbs-charn_remapped_1d_ivt.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo319-DART-ssp585d-gibbs-charn_atmos_remapped_1d_ivt/ \
                          --vars ivt \
                          --pattern "Re_{var}_Global_AWI-CM3_SSP585_*.nc" \
                          --time-var "time"

###################################
# TCo1279-DART-1950C
##################################

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-1950C/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo1279-DART-1950C_3h.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo1279-DART-1950C_atmos_3h/ \
                          --vars 10u 10v 2t cp hcc lcc lsp mcc pl_w_850 tcc \
                          --pattern "atm_reduced_3h_{var}_3h_*.nc" \
                          --time-var "time_counter"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-1950C/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo1279-DART-1950C_remapped_1m.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo1279-DART-1950C_atmos_remapped_1m/ \
                          --vars ssrc  ssrd strc strd tsrc ttr ttrc fal\
                          --pattern "atm_remapped_1m_{var}_1m_*.nc" \
                          --time-var "time_counter"

## Still have to do it, as there are some missing/corrupted data.
python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-1950C/outdata/fesom \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo1279-DART-1950C_ocean_native_daily_3d.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo1279-DART-1950C_ocean_native_daily_3d/ \
                          --vars salt1-31 temp1-31  \
                          --pattern "{var}.fesom.*.nc" \
                          --time-var "time"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-1950C/outdata/fesom \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo1279-DART-1950C_ocean_native_daily_3d_elements.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo1279-DART-1950C_ocean_native_daily_3d_elements/ \
                          --vars u1-31 v1-31 w1-31 \
                          --pattern "{var}.fesom.*.nc" \
                          --time-var "time"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-1950C/outdata/fesom \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo1279-DART-1950C_ocean_native_daily_2d.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo1279-DART-1950C_ocean_native_daily_2d/ \
                          --vars a_ice m_ice uice vice \
                          --pattern "{var}.fesom.*.nc" \
                          --time-var "time"


 python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-1950C/outdata/fesom \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo1279-DART-1950C_ocean_native_monthly.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo1279-DART-1950C_ocean_native_monthly/ \
                          --vars fh MLD2 tx_sur ty_sur temp salt \
                          --pattern "{var}.fesom.*.nc" \
                          --time-var "time"                         
###################################
# TCo1279-DART-2080C
##################################

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-2080C/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo1279-DART-2080C_remapped_1m.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo1279-DART-2080C_atmos_remapped_1m/ \
                          --vars ssrc  ssrd strc strd tsrc ttr ttrc fal\
                          --pattern "atm_remapped_1m_{var}_1m_*.nc" \
                          --time-var "time_counter"


python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-2080C/outdata/oifs \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo1279-DART-2080C_3h.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo1279-DART-2080C_atmos_3h/ \
                          --vars 10u 10v 2t cp hcc lcc lsp mcc pl_w_850 tcc \
                          --pattern "atm_reduced_3h_{var}_3h_*.nc" \
                          --time-var "time_counter"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-2080C/outdata/fesom \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo1279-DART-2080C_ocean_native_daily_3d.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo1279-DART-2080C_ocean_native_daily_3d/ \
                          --vars salt1-31 temp1-31 u1-31 v1-31 w1-31 \
                          --pattern "{var}.fesom.*.nc" \
                          --time-var "time"

python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-2080C/outdata/fesom \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo1279-DART-2080C_ocean_native_daily_2d.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo1279-DART-2080C_ocean_native_daily_2d/ \
                          --vars a_ice m_ice uice vice \
                          --pattern "{var}.fesom.*.nc" \
                          --time-var "time"

 python build_kern_catalog.py \
                          --data-dir /work/ab0995/ICCP_AWI_hackthon_2025/TCo1279-DART-2080C/outdata/fesom \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/TCo1279-DART-2080C_ocean_native_monthly.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_TCo1279-DART-2080C_ocean_native_monthly/ \
                          --vars fh MLD2 tx_sur ty_sur temp salt \
                          --pattern "{var}.fesom.*.nc" \
                          --time-var "time"  

############
# UHR ocean N39
############

python build_kern_catalog.py \
                          --data-dir /work/uo0119/a270067/runtime/awicm3-v3.1/N39/outdata/fesom/ \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/UHR_ocean_ocean_N39_daily.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_UHR_ocean_ocean_N39_daily/ \
                          --vars a_ice m_ice s050 s100 ssh sss sst t050 t100 tx_sur ty_sur uice unod00 unod100 unod30 vice vnod00 vnod100 vnod30 \
                          --pattern "{var}.fesom.*.nc" \
                          --time-var "time" \
                          --skip-first 1
# s050 s100 t050 t100 unod00 unod100 unod30 vnod00 vnod100 vnod30 

python build_kern_catalog.py \
                           --data-dir /work/uo0119/a270067/runtime/awicm3-v3.1/N39/outdata/fesom/ \
                           --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/UHR_ocean_ocean_N39_daily_surface.json \
                           --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_UHR_ocean_ocean_N39_daily_surface/ \
                           --vars a_ice m_ice ssh sss sst tx_sur ty_sur uice vice \
                           --pattern "{var}.fesom.*.nc" \
                           --time-var "time" \
                           --skip-first 1

python build_kern_catalog.py \
                          --data-dir /work/uo0119/a270067/runtime/awicm3-v3.1/N39/outdata/fesom/ \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/UHR_ocean_ocean_N39_monthly.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_UHR_ocean_ocean_N39_monthly/  \
                          --vars alb ist MLD1 MLD2 MLD3 momix_length m_snow N2 qcon qres qsi salt temp unod vnod w \
                          --pattern "{var}.fesom.*.nc"\
                          --time-var "time"\
                          --skip-first 1

############
# UHR ocean N43
############

python build_kern_catalog.py \
                          --data-dir /work/uo0119/a270067/runtime/awicm3-v3.1/N43/outdata/fesom/ \
                          --out-json /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/UHR_ocean_ocean_N43_monthly.json \
                          --per-var-dir /work/ab0995/ICCP_AWI_hackthon_2025/kerchunk/per_var_UHR_ocean_ocean_N43_monthly/  \
                          --vars temp alb ist MLD1 MLD2 MLD3 momix_length m_snow N2 qcon qres qsi salt unod vnod w \
                          --pattern "{var}.fesom.*.nc"\
                          --time-var "time"\
                          --skip-first 1