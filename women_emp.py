import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import seaborn as sns
from pywaffle import Waffle
import pyreadstat
from kmodes.kmodes import KModes
from sklearn.preprocessing import MinMaxScaler
from math import pi

import warnings
warnings.filterwarnings('ignore')

# Read the dataset
relevant_df = pd.read_csv('data/sprint1-clean.csv')

# drop the respondent who answered Don't know in education variables
relevant_df = relevant_df[relevant_df['V106'] != "Don't know"]

shapefile = gpd.read_file('data/geo/provinces/Provinces.shp')
shapefile["x"] = shapefile.geometry.centroid.x
shapefile["y"] = shapefile.geometry.centroid.y
    
df_recode = {'Isabela City' : 'Basilan',
             'Cotabato City': 'Maguindanao', 
             'Caloocan/Malabon/Navotas/Valenzuela': 'Metropolitan Manila',
             'Las Pinas/Makati/Muntinlupa/Paranaque/Pasay/Taguig/Pateros': 'Metropolitan Manila',
             'Mandaluyong/Marikina/Pasig/San Juan/Quezon City': 'Metropolitan Manila',
             'Manila': 'Metropolitan Manila',
             'Cebu (Inc Cities)':'Cebu',
             'Samar (Western)':'Samar',
             'Compostella Valley':'Compostela Valley',
             'Cotabato (North)':'North Cotabato'}

shp_recode = { 'Shariff Kabunsuan': 'Davao Occidental'}

# recode shapefile PROVINCE 
shp_recode = { 'Shariff Kabunsuan': 'Davao Occidental'}
shapefile['PROVINCE_'] =  shapefile['PROVINCE'].apply(lambda x: x.title()).replace(shp_recode)

# Recode SPROV
df_recode = {'Isabela City' : 'Basilan',
 'Cotabato City': 'Maguindanao',
 'Caloocan/Malabon/Navotas/Valenzuela': 'Metropolitan Manila',
 'Las Pinas/Makati/Muntinlupa/Paranaque/Pasay/Taguig/Pateros': 'Metropolitan Manila',
 'Mandaluyong/Marikina/Pasig/San Juan/Quezon City': 'Metropolitan Manila',
 'Manila': 'Metropolitan Manila',
 'Cebu (Inc Cities)':'Cebu',
 'Samar (Western)':'Samar',
 'Compostella Valley':'Compostela Valley',
 'Cotabato (North)':'North Cotabato'
}


relevant_df['SPROV_'] = relevant_df['SPROV'].apply(lambda x: x.title()).replace(df_recode)

# page nav
my_page = st.sidebar.radio('Page Navigation', ['Dataset', 'Indicators', 'Profiles']) 


if my_page == 'Dataset':
    
    st.title("Women Empowerment")
    st.header("Data from 2017 Philippines Standard DHS")
    if st.checkbox('Show data', value = True):
        st.subheader('Data')
        data_load_state = st.text('Loading data...')
        st.write(relevant_df.head(20))
        data_load_state.markdown('Loading data...**done!**')
        
    st.subheader("Objectives:")
    st.write("1. Give a glimpse of the state of women empowerment in the Philippines.")
    st.write("2. Identify areas of concern with regards to women empowerment in the Philippines.") 
    st.write("3. Suggest government interventions to address areas of concern with regard to women empowerment in the Philippines. ")
    
elif my_page == 'Indicators':
    
    st.header("Ability to Negotiate Sexual Relationships")
    
    # Ability to Negotiate Sexual Relationships
    sexrel_df = relevant_df[['CASEID', 'V012', 'V013', 'V212', 'V024', 'V102', 'SPROV', 'V106', 'V133', 
                             'V190', 'V170', 'V502', 'V633B', 'V822', 'V850A', 'V850B']]
    sexrel_df.groupby(['V190', 'V850A'])['V850A'].size()
    sexrel_a = sexrel_df.groupby(['V190', 'V850A']).agg(size=("V850A", "size")).reset_index()
    sexrel_b = sexrel_df.groupby(['V190', 'V850B']).agg(size=("V850B", "size")).reset_index()
    
    fig1 = plt.figure(figsize=(12,15))
    #1 row 2 columns

    # bar 1
    ax1 = plt.subplot2grid((2,1),(0,0))
    sns.reset_defaults()

    distribution = pd.crosstab(sexrel_df['V190'], sexrel_df['V850A'], normalize='index')

    label = ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest']

    # plot the cumsum, with reverse hue order
    sns.barplot(data=distribution.cumsum(axis=1).stack().reset_index(name='Dist'),
            x=sexrel_a['V190'], y='Dist', hue=sexrel_a['V850A'],
            hue_order = distribution.columns[::-1],   # reverse hue order so that the taller bars got plotted first
            order=label, dodge=False, palette = 'tab10')
    plt.xlabel("Wealth Quintile", fontsize=12)
    # plt.ylabel("Distribution", fontsize=12)
    plt.legend(title="Able to Refuse Sex", fontsize=12)
    plt.yticks(fontsize=12)


    # bar 2
    ax1 = plt.subplot2grid((2,1), (1,0))
    sns.reset_defaults()

    distribution = pd.crosstab(sexrel_df['V190'], sexrel_df['V850B'], normalize='index')

    label = ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest']

    # plot the cumsum, with reverse hue order
    sns.barplot(data=distribution.cumsum(axis=1).stack().reset_index(name='Dist'),
            x=sexrel_b['V190'], y='Dist', hue=sexrel_b['V850B'],
            hue_order = distribution.columns[::-1],   # reverse hue order so that the taller bars got plotted first
            order=label, dodge=False, palette = 'tab10')
    plt.xlabel("Wealth Quintile", fontsize=12)
    # plt.ylabel("Distribution", fontsize=12)
    plt.legend(title="Ask Partner to Wear Condom", fontsize=12)
    plt.yticks(fontsize=12)
    
    st.pyplot(fig1)
    
    st.subheader("Insights:")
    st.write("Filipino women who are currently in union are able to say no to their husbands if they do not want to have sexual intercourse. Most are also able to ask their husbands to wear a condom. True across all wealth quintiles.")
    
    #Household decision-making
    decision_df = relevant_df[['CASEID', 'V012', 'V013', 'V212', 'V024', 'V102', 'SPROV', 'V106', 'V133', 
                           'V190', 'V170', 'V502', 'V731', 'V739','V743A', 'V743B', 'V743D', 'V743F']]
    
    wife_earn = decision_df.groupby("V739").size().sort_values(ascending=False).head(3)
    health = decision_df.groupby('V743A').size().sort_values(ascending=False).head(3)
    large_purch = decision_df.groupby('V743B').size().sort_values(ascending=False).head(3)
    visits = decision_df.groupby('V743D').size().sort_values(ascending=False).head(3)
    husb_earn = decision_df.groupby('V743F').size().sort_values(ascending=False).head(3)
    
    st.header("Household Decision-Making")
    fig2 = plt.figure(figsize=(60,44))
    #1 row 2 columns

    # donut 1
    ax1 = plt.subplot2grid((1,5),(0,0))
    plt.pie(wife_earn, textprops={'fontsize': 12}, 
                        autopct='%1.1f%%', startangle=90, pctdistance=0.77)
    circle = plt.Circle((0,0), 0.55, color='white')
    p=plt.gcf()
    p.gca().add_artist(circle)
    plt.title('Respondent\'s Earnings')

    # donut 2
    ax1 = plt.subplot2grid((1,5), (0, 1))
    plt.pie(health, textprops={'fontsize': 12}, 
                        autopct='%1.1f%%', startangle=90, pctdistance=0.77)
    circle = plt.Circle((0,0), 0.55, color='white')
    p=plt.gcf()
    p.gca().add_artist(circle)
    plt.title('Own Healthcare')

    # donut 3
    ax1 = plt.subplot2grid((1,5), (0, 2))
    plt.pie(large_purch, textprops={'fontsize': 12}, 
                        autopct='%1.1f%%', startangle=90, pctdistance=0.77)
    circle = plt.Circle((0,0), 0.55, color='white')
    p=plt.gcf()
    p.gca().add_artist(circle)
    plt.title('Large Household Purchases')

    # donut 4
    ax1 = plt.subplot2grid((1,5), (0, 3))
    plt.pie(visits, textprops={'fontsize': 12}, 
                        autopct='%1.1f%%', startangle=90, pctdistance=0.77)
    circle = plt.Circle((0,0), 0.55, color='white')
    p=plt.gcf()
    p.gca().add_artist(circle)
    plt.title('Visits to Relatives')

    # donut 5
    ax1 = plt.subplot2grid((1,5), (0, 4))
    plt.pie(husb_earn, textprops={'fontsize': 12}, 
                        autopct='%1.1f%%', startangle=90, pctdistance=0.77)
    circle = plt.Circle((0,0), 0.55, color='white')
    p=plt.gcf()
    p.gca().add_artist(circle)
    plt.title('Husband\'s Earnings')
    
    st.pyplot(fig2)
    
    st.subheader("Insights:")
    st.write("Filipino women who are currently in union are able to say no to their husbands if they do not want to have sexual intercourse. Most are also able to ask their husbands to wear a condom. True across all wealth quintiles.")
    
    st.header("Attitude Towards Wife-Beating")

    dict_users = {'No': 87, 'Yes in at least one': 12, 'Yes to all': 1}
    wifebeat_df = pd.Series(dict_users)
    
    fig3 = plt.figure(FigureClass=Waffle, figsize=(5,5), values=wifebeat_df, rows=10, icons='user')
    
    st.pyplot(fig3)
    
    # by province
    st.subheader("Geospatial Analysis on Women Who Agree to at Least 1 Reason")
    beating_df = relevant_df[['CASEID', 'V012', 'V013', 'V212', 'V024', 'V102', 'SPROV_', 'V106', 'V133', 
                           'V190', 'V170', 'V502', 'V744A', 'V744B', 'V744C', 'V744D', 'V744E']]
    
    beating_df.reset_index(drop=True)
    
    agree = beating_df[(beating_df['V744A']=="Yes") | 
           (beating_df['V744B']=="Yes") |
           (beating_df['V744C']=="Yes") |
           (beating_df['V744D']=="Yes") |
           (beating_df['V744E']=="Yes")
          ]
    shp_list = [x.title() for x in shapefile["PROVINCE"].unique()]
    df_list = [x.title() for x in agree["SPROV_"].unique()]
    
    province_df = agree.groupby('SPROV_', as_index = False).agg(size=('CASEID','size'))

    merged_data = pd.merge(shapefile, province_df, left_on = 'PROVINCE_', right_on = 'SPROV_')

    variable1 = 'size'
    vmin, vmax = merged_data[variable1].min(), merged_data[variable1].max()

    fig4, ax = plt.subplots(1, figsize=(25, 15))

    merged_data.plot(column=variable1, cmap='magma_r', linewidth=0.8, ax=ax, edgecolor='0.8', vmin=vmin, vmax=vmax)

    plt.xlim(115,130)
    plt.ylim(0,25)
    # plt.title("Number of Teenage Mom Respondents by Province", fontsize = 16)

    sm = plt.cm.ScalarMappable(cmap='magma_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig4.colorbar(sm)

    st.pyplot(fig4)
    
    st.subheader("Insights:")
    st.write("Respondents were asked whether they agree to the following reasons justifying wife beating:  argues with husband, neglects the children, goes out without telling husband, refuses to have sex with husband, and/or burns the food. 87% of the respondents living with a partner disagree with all of the specific reasons justifying wife-beating. Notable concentrations can be found in the following regions: ARMM, Western Visayas, Zamboanga Peninsula, Cordillera, and Davao.")
    
elif my_page == 'Profiles':
    
    st.title("Cluster Profiling of Women based on the Empowerment Indicators")
    
#     # load dataset for clustering
#     df, meta = pyreadstat.read_sav('data/PHIR71FL.SAV')
    
#     #converting variables into 0 and 1, 1 if pro human empowerment
#     for k,v in meta.variable_value_labels.items():
#         #ownership
#         if "V743" in k:
#             print(k, v)
#             df[k+ "_"] = df[k].map({1: 1, 2: 1,3:1,4:0,5:0,6:0})
#         #beating, coding reversed since 1 is pro beating in initial data
#         elif "V744" in k:
#             print(k,v)
#             df[k+ "_"] = df[k].map({0: 1, 1:0})
#         elif ("V745A" in k) | ("V745B" in k):
#             print(k,v)
#             df[k+ "_"] = df[k].map({1: 1, 0:0, 2:1,3:1})
            
#     # Wife_beating var 
#     #Adding All Beating Scores
#     df["Beating"] = df["V744A_"].fillna(0) + df["V744B_"].fillna(0) + df["V744C_"].fillna(0) + df["V744D_"].fillna(0) + df["V744E_"].fillna(0)

#     #1 if doesn't favor any beating else 0
#     df["wife_beating"] = np.where(df["Beating"] == 5, 1, 0)
    
#     # household decisions var
#     #Adding All Household descisions
#     df["HH_Dec"] = df["V743A_"] + df["V743B_"]  + df["V743D_"] + df["V743F_"]

#     # 1 if has household decisions 0 if none
#     df["household_dec"] = np.where(df["HH_Dec"] > 0, 1, 0)

#     # property ownership var
#     #Adding All Property Ownership
#     df["Property_Own"] = df["V745A_"] + df["V745B_"]

#     # 1 if owns property 0 if none
#     df["property_owner"] = np.where(df["Property_Own"] > 0, 1, 0)
    
#     # educational attainment
#     df["educated"] = np.where((df["V149"] == 4) | (df["V149"]==5), 1, 0)
    
#     # can refuse sex
#     df["can_refuse_sex"] = np.where((df["V633B"] == 1) | (df["V822"]==1), 1, 0)
    
#     # knows contraceptoion
#     df["knows_contraception"] = np.where((df["V301"] > 0), 1, 0)
    
#     # employed
#     df["has_work"] = np.where((df["V731"] > 0), 1, 0)
    
#     #Metrics for women empowerment
#     to_kmodes_df = df[["wife_beating", "household_dec","educated","can_refuse_sex","knows_contraception","has_work"]]
    
#     # kmodes clusters = 5
#     km = KModes(n_clusters=5, init='random', n_init=5, verbose=1, random_state = 42)
#     cluster_labels = km.fit_predict(to_kmodes_df)
    
#     to_kmodes_df['cluster_labels'] = cluster_labels
#     temp2_df = to_kmodes_df.copy()
    
#     # prepare for plot
#     scaler = MinMaxScaler()

#     #Remove c - color column from df since it's not needed for this visuals
#     #Drop Cluster labels for now since we don't want to scale its values 
#     temp2_df.drop(columns = [ 'cluster_labels'], inplace = True)

#     df_minmax = scaler.fit_transform(temp2_df)
#     df_minmax = pd.DataFrame(df_minmax, index=temp2_df.index, columns=temp2_df.columns)

#     df_minmax['cluster_labels'] = cluster_labels

#     df_clusters = df_minmax.set_index("cluster_labels")
#     df_clusters = df_clusters.groupby("cluster_labels").mean().reset_index().fillna(0)

#     ##### SPIDER PLOT #####
#     def make_spider( row, title, color):
 
#         # number of variable
#         categories=list(df_clusters)[1:]
#         N = len(categories)

#         # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
#         angles = [n / float(N) * 2 * pi for n in range(N)]
#         angles += angles[:1]

#         # Initialise the spider plot
#         ax = plt.subplot(3,3,row+1, polar=True )

#         # If you want the first axis to be on top:
#         ax.set_theta_offset(pi / 3.5)
#         ax.set_theta_direction(-1)

#         # Draw one axe per variable + add labels labels yet
#         plt.xticks(angles[:-1], categories, color='grey', size=8)

#         # Draw ylabels
#         ax.set_rlabel_position(0)
#         plt.yticks([-0.25, 0, 0.25, 0.5, 0.75, 1], [-0.25, 0, 0.25, 0.5,0.75, 1], color="grey", size=7) #formmscaled
#         plt.ylim(-0.25,1)

#         # Ind1
#         values=df_clusters.loc[row].drop('cluster_labels').values.flatten().tolist()
#         values += values[:1]
#         ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
#         ax.fill(angles, values, color=color, alpha=0.4)

#         # Add a title
#         plt.title(title, size=14, color=color, y=1.1)
    
#     my_dpi=100
#     fig5 = plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
#     plt.subplots_adjust(hspace=0.5)

#     # Create a color palette:
#     my_palette = plt.cm.get_cmap("Set2", len(df_clusters.index))

#     for row in range(0, len(df_clusters.index)):
#         make_spider(row=row, 
#                 title='Segment '+(df_clusters['cluster_labels'][row]).astype(str), 
#                 color=my_palette(row))
    
#     st.pyplot(fig5)
    
    st.header("Segments:")
    
    st.subheader("Segment 0. Empowered Cluster")
    st.write("35% of Women Fall under this cluster. Generally, women in this cluster are educated and are employed. They participate in the house-hold decision making process, disagree to all specific reasons for justifying wife-beating, and are able to refuse sexual intercourse. 60% of Women under this cluster owns a property.")
    
    st.subheader("Segment 1. Generally Empowered But Jobless")
    st.write("30% of Women fall under this cluster. Women in this cluster participate in the house-hold decision making process, disagree to all specific reasons for justifying wife-beating, and are able to refuse sexual intercourse. However, they are lack empowerment in terms of employment.") 
    
    st.subheader("Segment 2. Empowered but Submissive in Household")
    st.write("18% of Women fall under this cluster. Although the women in this cluster disagree to all specific reasons for justifying wife-beating and are able to refuse sexual intercourse, they hardly participate in the household decision-making process.")
    
    st.subheader("Segment 3. Generally Lacks Empowerment")
    st.write("7% of Women fall under this cluster. Women in this cluster do not exhibit empowerment in terms of employment, education, and participation in the household decision. They are also not able to refuse sexual intercourse and, to some extent, agree to the reasons for wife-beating.")
    
    st.subheader("Segment 4. Only Sexually Empowered")
    st.write("11% of Women fall under this  Cluster. This cluster consists of women who know about contraception and are empowered enough to refuse sex. This cluster also includes women who disagree to all specific reasons for justifying wife-beating.")
    
    

    