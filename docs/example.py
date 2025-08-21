def run_causal_analysis(df, asset, start, end):
    start = pd.to_datetime(start)
    end=pd.to_datetime(end)
    df['timerange'] = pd.to_datetime(df['timerange'])
    df = df[(df['timerange'] > start) & (df['timerange'] < end)]
   
    alarm_downtime_conn = [(col, 'Downtime') for col in asset.cols]
    alarm_lags_downtime_conn = [(col, 'Downtime') for col in asset.lags]
    alarm_lags = [(col +'_lag1', col) for col in asset.cols]
    alarm_lags2 = [(col +'_lag2', col+'_lag1') for col in asset.cols]
    alarm_lags3 = [(col+'_lag3', col +'_lag2') for col in asset.cols]
 
    causal_graph = nx.DiGraph([*alarm_downtime_conn, *alarm_lags_downtime_conn, *alarm_lags, *alarm_lags2, *alarm_lags3])
 
    pos = nx.spring_layout(causal_graph)
    nx.draw(causal_graph, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=8, font_weight='bold', edge_color='gray')
    plt.show()
 
    # Binarization
    for col in asset.cols:
        df[col] = df[col].apply(lambda x: 0 if x < 1.0 else 1)
    for col in asset.lags:
        df[col] = df[col].apply(lambda x: 0 if x < 1.0 else 1)
 
 
    df = df.drop(columns=[col for col in df.columns if col not in asset.cols + asset.lags + ["Downtime"]])
 
    results = []
    for alarm in asset.cols:
        logger.info(f"Analyzing causal effect of {alarm} on Output...")
       
        try:
            model = CausalModel(
                data=df,
                treatment=alarm,
                outcome='Downtime',
                graph=causal_graph
            )
           
            # Identify the causal effect
            identified_estimand = model.identify_effect()
 
            if args.method == "backdoor.generalized_linear_model":
                method_params = {"glm_family": statsmodels.api.families.Binomial()}
            else:
                method_params = None
           
            # Estimate the causal effect
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=args.method,
                method_params=method_params
            )
           
            # Refute the estimate
            refutation = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="random_common_cause"
            )
           
            # Store the results
            results.append({
                'Alarm': alarm,
                'Causal_Effect': estimate.value,
                'Refutation_Pvalue': refutation.refutation_result["p_value"],
                'Refutation_Passed': refutation.refutation_result["is_statistically_significant"]
            })
 
            with open(os.path.join(args.out, f"causal_model_{alarm}.pkl"), "wb") as file:
                pickle.dump(model, file)
            with open(os.path.join(args.out, f"estimand_{alarm}.pkl"), "wb") as file:
                pickle.dump(identified_estimand, file)
       
        except Exception as e:
            logger.error(f"Error analyzing {alarm}: {e}")
            results.append({
                'Alarm': alarm,
                'Causal_Effect': None,
                'Refutation_Pvalue': None,
                'Refutation_Passed': False
            })
 
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.out, f"results_{asset.name}.csv"))
 
 
if __name__ == "__main__":
    tbl = mltable.load(args.data)
    df = tbl.to_pandas_dataframe()
 
    assets = ["Cartoner", "CasePacker", "Labeller", "TUnload", "Tamper", "TraySealer"]
    assets_dict = {}
    for asset in assets:
        assets_dict[asset] = AssetData(df, asset)
    df = df.fillna(0)
 
    if args.asset != "all":
        if args.asset == "Cartonner":
            asset = assets_dict["Cartoner"]
        else:
            asset = assets_dict[args.asset]
        run_causal_analysis(df=df[df["RootCauseL2"] == args.asset], asset=asset, start=" ".join(args.start.split("_")), end = " ".join(args.end.split("_")))
    else:
        # partial_function = partial(run_causal_analysis, df=df)
 
        with Pool(processes=4) as pool:
            results = pool.starmap(run_causal_analysis, [(df, assets_dict[asset]) for asset in assets])