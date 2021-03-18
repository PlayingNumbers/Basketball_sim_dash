# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:07:58 2021

@author: kenne
"""
import pandas as pd 
import random as rd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import streamlit as st 

# build team class for simulation
class Team:
    def __init__(self, teamid, data, season):
        self.teamid = teamid
        self.data = data[(data['TeamID'] == self.teamid)&
                         (data['Season'] == season)].copy()
        self.team_name = self.data['TeamName'].unique()[0]
    
    def getPointsScored(self):
        return self.data['PtScored'].values

    def getPointsAllowed(self):
        return self.data['PtAllowed'].values
    
    def getAttributes(self):
        self.attributes = dict()
        for col in self.data.columns:
            self.attributes[col] = self.data[col].values
        return self.attributes
    
# make simulation functions

def sim_once(team1,team2):
    score_team1= rd.gauss(team1.getPointsScored().mean(),team1.getPointsScored().std())
    score_team2= rd.gauss(team2.getPointsScored().mean(),team2.getPointsScored().std())
    score_against_team1= rd.gauss(team1.getPointsAllowed().mean(),team1.getPointsAllowed().std())
    score_against_team2= rd.gauss(team2.getPointsAllowed().mean(),team2.getPointsAllowed().std())
    final_score_t1 = (score_team1+score_against_team2)/2
    final_score_t2 = (score_team2+score_against_team1)/2
    if final_score_t1 == final_score_t2:
        sim_once(team1,team2)
    return (final_score_t1,final_score_t2, final_score_t1 > final_score_t2)
    

def sim_multiple(team1,team2,n=100):
    """Takes two teams in and returns win % of t1, t1 point dist, t2 point dist, win loss binary"""
    t1_points = []
    t2_points = []
    w_l = []
    for i in range(n):
        sim = sim_once(team1,team2)
        t1_points.append(sim[0])
        t2_points.append(sim[1])
        w_l.append(sim[2])
    return (sum(w_l)/n, t1_points, t2_points, w_l)



@st.cache
def load_data():
    by_team = pd.read_csv("sim_data_out.csv")
    return by_team

df = load_data()

st.title('NCAAB Matchup Simulator Tool')
seasons = df.Season.unique()
seasons.sort()
year = st.selectbox('Year:', seasons)

t1 = st.selectbox('Team 1:', df[df['Season'] == year].TeamName.unique())
t2 = st.selectbox('Team 2:', df[df['Season'] == year].TeamName.unique())

t1_id = df[df.TeamName == t1]['TeamID'].unique()[0]
t2_id = df[df.TeamName == t2]['TeamID'].unique()[0]

team_1 = Team(t1_id,df,year)
team_2 = Team(t2_id,df,year)

sim_out = sim_multiple(team_1,team_2,1000)

st.write(t1, ' Win Probability:', sim_out[0])


fig, ax = plt.subplots(1)
sns.kdeplot(sim_out[1], label = t1, ax=ax)
sns.kdeplot(sim_out[2], label = t2, ax=ax)
fig.suptitle("Projected Points Scored Distribution")
ax.legend()
st.pyplot(fig)