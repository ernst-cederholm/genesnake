import dash
from dash import Dash, html, dcc, callback, callback_context, Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# function for running everything
def interactive_plots(data_df, rep_dict):
    """
    Makes a multipage dash app with interactive figures for more detailed viewing
    INPUT:
        data_df - a column sorted pandas.DataFrame with labled gene expression data, genesXexperiments
        rep_dict - a dictionary with experiment/gene names as keys and values as a list of indecies for which experiments are replicates of eachother
    """
    # initilize a dash app with pages, where the pages will be found within the same document
    app = Dash(__name__, use_pages=True, pages_folder="")

    # sort the rown for nicer heatmaps
    data_df = data_df.sort_index(axis=0)

    def pca_fit_and_format(data_df, rep_dict, direct):
        """
        Function for fitting a two component PCA to gene expression
        INPUT:
            data_df - a sorted pandas.DataFrame with labled gene expression data, genesXexperiments
            rep_dict - a dictionary with experiment/gene names as keys and values as a list of indecies for which experiments are replicates of eachother
            direct - tag for if the PCA should be done over the experiments or genes
        OUTPUT:
            component_df - a dataframe with the principal components of the PCA and one to mark which replicate it is
        """
        pca = PCA(n_components=2)

        if direct=="experiments":
            # find principal components for observations
            pca_components = pca.fit_transform(np.transpose(data_df.values))

            # make list for which replicate observations are part of
            group_markers = np.empty(data_df.shape[1])
            for i in rep_dict.keys(): # loop over experiments
                group = 0
                obs_inds = rep_dict[i]
                for j in obs_inds: # loop over replicates for experiment
                    group_markers[j] = group
                    group += 1
        else:
            # fit PCA
            pca_components = pca.fit_transform(data_df.values)
            group_markers = np.zeros(data_df.shape[0])

        component_df = pd.DataFrame(data=pca_components, columns=["Principal component 1", "Principal component 2"])
        component_df["Group"] = group_markers

        # set the correct index depending on which direction it has been run
        if direct=="experiments":
            component_df.index = np.concatenate([[key]*len(value) for key, value in rep_dict.items()])
        else:
            component_df.index = data_df.index

        return component_df

    # PCA over experiments, choose experiment to highlight
    pca_exp_components = pca_fit_and_format(data_df, rep_dict, "experiments")

    # layout of PCA over experiments where you can pick which experiment to focus on with a dropdown menu
    pca_exp_choose_gene_layout = html.Div([
            html.H2('PCA over experiments'),
            html.P("Select experiment/perturbed gene:"),
            dcc.Dropdown(
                id="pca_dropdown",
                options=pca_exp_components.index.unique(),
                value=pca_exp_components.index[0],
                clearable=False,
            ),
            dcc.Loading(dcc.Graph(id="pca_exp_graph"))
            ])

    # function to update the figure depending on experiment chosen
    @callback(
        Output("pca_exp_graph", "figure"),
        Input("pca_dropdown", "value"))
    def highlight_experiment(exp):
        fig = go.Figure()
        chosen = pca_exp_components[pca_exp_components.index==exp]
        other = pca_exp_components[pca_exp_components.index!=exp]
        fig.add_trace(go.Scatter(x=other["Principal component 1"], y=other["Principal component 2"], mode='markers', text=other.index, name=""))
        fig.add_trace(go.Scatter(x=chosen["Principal component 1"], y=chosen["Principal component 2"], mode='markers', text=chosen.index, name=""))
        fig.update_layout(showlegend=False, title_text="2 component PCA of experiments", title_subtitle_text="Chosen experiment group marked in red")
        return fig
    
    # PCA over experiments animation layout where you can pick if the animation should highlight experiments or replicates
    pca_exp_anim_layout = html.Div([
                    html.H2('PCA over experiments'),
                    dcc.Loading(dcc.Graph(id="animation_graph")),
                    dcc.RadioItems(id="animation_type", options=["Experiment", "Replicate"], value="Replicate", inline=True)
                    ])

    # Function to update which animation to run
    @callback(
        Output("animation_graph", "figure"),
        Input("animation_type", "value")
    )
    def animation(type):
        # Set up base figure
        max_reps = np.max([len(reps) for reps in rep_dict.values()])
        pca_exp_anim_fig = go.Figure(data=[go.Scatter(x=pca_exp_components["Principal component 1"], 
                                            y=pca_exp_components["Principal component 2"], 
                                            mode='markers', text=pca_exp_components.index, name=""),
                                        go.Scatter(x=pca_exp_components["Principal component 1"][pca_exp_components["Group"]==0], 
                                            y=pca_exp_components["Principal component 2"][pca_exp_components["Group"]==0],
                                            mode='markers', text=pca_exp_components[pca_exp_components["Group"]==0].index, name="")])
        # Update layout with play and pause buttons for the animation
        pca_exp_anim_fig.update_layout(title_text="Animation of PCA over experiments", showlegend=False,
                updatemenus = [dict(type = "buttons",
                buttons = [
                    dict(args = [None],
                        label = "Play",
                        method = "animate"
                        ),
                    dict(args = [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                        label = "Pause",
                        method = "animate")])])
        # Depending on which animation chosen loop over correct group of points, updates the second scatter plot of the ones initially drawn
        if type == "Replicate":
            pca_exp_anim_fig.update(frames=[go.Frame(
                    data=[go.Scatter(x=pca_exp_components["Principal component 1"][pca_exp_components["Group"]==k], 
                                                        y=pca_exp_components["Principal component 2"][pca_exp_components["Group"]==k],
                                                        mode='markers', text=pca_exp_components[pca_exp_components["Group"]==k].index, name="")],
                            traces=[1])
                            for k in range(max_reps)])
        else:
            pca_exp_anim_fig.update(frames=[go.Frame(
                    data=[go.Scatter(x=pca_exp_components["Principal component 1"][pca_exp_components.index==exp], 
                                                        y=pca_exp_components["Principal component 2"][pca_exp_components.index==exp],
                                                        mode='markers', text=pca_exp_components[pca_exp_components.index==exp].index, name="")],
                            traces=[1])
                            for exp in pca_exp_components.index.unique()])
        return pca_exp_anim_fig
            
    # PCA over genes
    pca_gene_components = pca_fit_and_format(data_df, rep_dict, direct="genes")

    # make basic figure
    pca_gene_fig = go.Figure(go.Scatter(x=pca_gene_components["Principal component 1"], y=pca_gene_components["Principal component 2"], mode='markers', text=pca_gene_components.index, name=""))
    pca_gene_fig.update_layout(showlegend=False, title_text="2 component PCA of genes")

    # set the layout
    pca_gene_layout = html.Div([
                html.H2('PCA over genes'),
                dcc.Graph(figure=pca_gene_fig),
                ])

    # Heatmap
    # Check if there are enough experiments matched with genes to give the option to only show a subset of them
    overlap = list(set(rep_dict.keys())&set(data_df.index))
    if len(overlap)>len(rep_dict.keys())*0.6:
        radio = dcc.RadioItems(id="gene_checked", options=["All genes", "Perturbed in experiments"], value="All genes", inline=True)
    else:
        radio = dcc.RadioItems(id="gene_checked", options=[{"label": "All genes"}, {"label": "Perturbed in experiments", 'disabled': True}], value="All genes", inline=True)

    # Make unique versions of the column lables if they are not by adding a number for the replicate
    if len(data_df.columns.unique()) < len(data_df.columns):
        x_labs = []
        for inds in rep_dict.values():
            x_labs.append([data_df.columns[inds[0]]+"_"+str(i) for i in inds])
        x_labs = np.concatenate(x_labs)
    else:
        x_labs = data_df.columns

    # layout of the heatmap which checkboxes for which experiments (and genes if applicabe) to show
    heatmap_layout = html.Div([
            html.H2("Heatmap"),
            dcc.Loading(dcc.Graph(id="heatmap_graph")),
            radio,
            dcc.Checklist(id="all_none", options=["All experiments"], value=["All experiments"], inline=True),
            dcc.Checklist(id="experiments_checked", options=list(rep_dict.keys()), value=list(rep_dict.keys())[:15], inline=True) # only first 15 are checked for faster loading on start
            ])

    # Function to show the correct version of the heatmap
    @callback(
        Output("all_none", "value"),
        Output("experiments_checked", "value"),
        Output("heatmap_graph", "figure"),
        Input("all_none", "value"),
        Input("experiments_checked", "value"),
        Input("gene_checked", "value"))
    def sync_heatmap(all_select, exp_select, gene_select):
        ctx = callback_context
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]
        # Update the checkboxes depending on which others are checked
        if input_id == "all_none":
            exp_select = list(rep_dict.keys()) if all_select else [list(rep_dict.keys())[0]]
        else:
            all_select = ["All experiments"] if set(exp_select) == set(rep_dict.keys()) else []

        # determine which experiments and genes to show
        exp_idx = np.concatenate([inds for key, inds in rep_dict.items() if key in exp_select])
        if gene_select == "Perturbed in experiments":
            genes = [gene for gene in data_df.index if gene in exp_select]
        else:
            genes = list(data_df.index)

        # replace all zeros with the closest value to avoid errors when logging
        min_val = np.nanmin(data_df.replace(to_replace=0, value=np.nan))
        z = data_df.replace(to_replace=0, value=min_val)
        max_val = np.max(data_df.iloc[:,exp_idx].loc[genes, :])

        # make the basic parts of the heatmap
        fig = go.Figure(go.Heatmap(z=np.log10(z.iloc[:,exp_idx].loc[genes, :].values), # log values so the color is readable
                                   x=x_labs[exp_idx], # set experiment lables
                                   y=data_df.loc[genes].index, # set gene lables
                                   customdata=data_df.iloc[:,exp_idx].loc[genes, :].values, # add back the data so correct values can be shown instead of logged
                                   name="",
                                   coloraxis='coloraxis',
                                   hovertemplate='expression:%{customdata}<br>experiment:%{x} <br>gene: %{y}' # customize the information on hover
                                   ))
        # redesign the colorbar with correct values and so you can see the color shift
        fig.update_coloraxes(colorscale="Inferno")
        fig.update_layout(coloraxis_colorbar=dict(
                    title = dict(text="Expression"), #come up with a better name
                    tickvals=np.log10([min_val, max_val/100, max_val/10, max_val]),
                    ticktext=["0", f"{(max_val/100):.2g}", f"{(max_val/10):.2g}", f"{max_val:.0f}"] 
                    ))
        # update layout so individual labels are not shown
        fig.update_layout(minreducedwidth=1000,
                        minreducedheight=2000,
                        xaxis=dict(showticklabels=False, title=dict(text="Experiments")), 
                        yaxis=dict(showticklabels=False, title=dict(text="Genes")))
        return all_select, exp_select, fig

    # Histogram
    plot_ray = data_df.values.flatten() # make 2D array 1D 
    bins = 100 #make % of total numbers as bins
    max_val = np.max(data_df) #largest value

    # define layout with slider to chose range to focus on
    histogram_layout = html.Div([
            html.H2("Histogram"),
            html.P("Select which parts of the histogram to focus on"),
            dcc.Loading(dcc.Graph(id="histogram")),
            dcc.RangeSlider(0, max_val, marks=None, value=[0, max_val], id="histogram_range")
            ])

    # Function to make figure with correct range
    @callback(
        Output("histogram", "figure"),
        Input("histogram_range", "value"))
    def histogram_for_range(range_vals):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=plot_ray, xbins=dict(start=range_vals[0], end=range_vals[1], size=(range_vals[1]-range_vals[0])/bins)))
        return fig

    # Landing page layout
    info_layout = html.Div([
                html.H2("Heatmap"),
                html.P("Log colored heatmap of the gene expression where you select which experiments and if all or only perturbed genes should be shown. Will add a replicate number to non-unique experiment labels."),
                html.H2("Histogram"),
                html.P("Distribution of the gene expression with a slider to pick which interval to focus on."),
                html.H2("PCA over experiments"),
                html.P("Two component PCA over the experiments where you select which experiment to highlight."),
                html.H2("PCA over experiments animation"),
                html.P("Two component PCA over the experiments with two animation options. One which highlights an experiment at a time, and one where the replicates of the same number are highlighted (primarily relevant for checking for batch effects)."),
                html.H2("PCA over genes"),
                html.P("Two component PCA over the genes.")
                ])

    # Register pages made
    dash.register_page("Overview", path="/", layout=info_layout) ## could use a better name
    dash.register_page("PCA_over_experiments", layout=pca_exp_choose_gene_layout)
    dash.register_page("PCA_over_experiments_animation", layout=pca_exp_anim_layout)
    dash.register_page("PCA_over_genes", layout=pca_gene_layout)
    dash.register_page("Histogram", layout=histogram_layout)
    dash.register_page("Heatmap", layout=heatmap_layout)

    # General app setup which will be visable at the top of each page
    app.layout = html.Div([
        html.H1('EDA interactive plots'),
        html.Div([
            html.Div(
                dcc.Link(f"{page['name']}", href=page["relative_path"])
            ) for page in dash.page_registry.values()
        ]),
        dash.page_container
    ])

    app.run(debug=True)