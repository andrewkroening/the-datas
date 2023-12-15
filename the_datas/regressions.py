def base_chart(data, yvar, xvar, plottitle, yname, xname, dcolor="grey"):
    chart = (
        alt.Chart(data)
        .mark_point(color=dcolor, opacity=0.5)
        .encode(
            x=alt.X(xvar, title=xname, scale=alt.Scale(zero=False)),
            y=alt.Y(yvar, title=yname, scale=alt.Scale(zero=False)),
        )
        .properties(title=plottitle)
    )
    return chart


def get_reg_fit(data, yvar, xvar, alpha=0.05, dcolor="grey"):

    # Grid for predicted values
    x = data.loc[pd.notnull(data[yvar]), xvar]
    xmin = x.min()
    xmax = x.max()
    step = (xmax - xmin) / 100
    grid = np.arange(xmin, xmax + step, step)
    predictions = pd.DataFrame({xvar: grid})

    # Fit model, get predictions
    model = smf.ols(f"{yvar} ~ {xvar}", data=data).fit()
    model_predict = model.get_prediction(predictions[xvar])
    predictions[yvar] = model_predict.summary_frame()["mean"]
    predictions[["ci_low", "ci_high"]] = model_predict.conf_int(alpha=alpha)

    # Build chart
    reg = (
        alt.Chart(predictions)
        .mark_line(color=dcolor)
        .encode(
            x=alt.X(xvar),
            y=alt.Y(yvar),
        )
    )
    ci = (
        alt.Chart(predictions)
        .mark_errorband(color=dcolor, opacity=0.2)
        .encode(
            x=xvar,
            y=alt.Y("ci_low"),
            y2="ci_high",
        )
    )
    chart = ci + reg

    return predictions, chart
