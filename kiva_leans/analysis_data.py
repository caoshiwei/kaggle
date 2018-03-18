#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/2 11:11
# @Author  : Cao Shiwei
# @File    : analysis_data.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_kiva = pd.read_csv("./data/kiva_loans.csv")
df_kiva_loc = pd.read_csv("./data/kiva_mpi_region_locations.csv")

print(df_kiva.shape)
print(df_kiva.nunique())
print(df_kiva.describe())
# df_kiva.head()

print("Distribution")
print(df_kiva[['funded_amount', 'loan_amount']].describe())

plt.figure(figsize=(12, 10))

plt.subplot(222)
g = sns.distplot(np.log(df_kiva['funded_amount'] + 1))
g.set_title("Funded Amount Distribution", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(221)
g1 = sns.distplot(np.log(df_kiva['loan_amount'] + 1))
g1.set_title("Loan Amount Distribution", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Count", fontsize=12)
plt.show()

lenders = df_kiva.lender_count.value_counts()
plt.figure(figsize=(12,10))

plt.subplot(222)
g = sns.distplot(np.log(df_kiva['lender_count'] + 1))
g.set_title("Dist of lenders log", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(221)
g1 = sns.distplot(df_kiva[df_kiva['lender_count'] < 1000]['lender_count'])
g1.set_title("Dist lender count")
g1.set_xlabel("")
g1.set_ylabel("count", fontsize=12)

plt.subplot(212)
g2 = sns.barplot(x=lenders.index[:40], y=lenders[:40])
g2.set_xticklabels(g2.get_xticklabels(), rotation=90)
g2.set_xlabel("")
g2.set_ylabel("Count", fontsize=12)
plt.show()

months = df_kiva.term_in_months.value_counts()
plt.figure(figsize=(12,10))

plt.subplot(222)
g = sns.distplot(df_kiva['term_in_months'])
g.set_title("Term in Months", fontsize=12)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(221)
g1 = sns.distplot(np.log(df_kiva['term_in_months'] + 1))
g1.set_title("Term in Months log", fontsize=12)
g1.set_xlabel("")
g1.set_ylabel("Count log", fontsize=12)

plt.subplot(211)
g2 = sns.barplot(x=months.index[:40], y=months.values[:40])
g2.set_xticklabels(g2.get_xticklabels(), rotation=90)
g2.set_title("the top 40 of term months", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("count", fontsize=12)
plt.show()

sector_amount = pd.DataFrame(df_kiva.groupby(['sector'])['loan_amount'].mean().
                             sort_values(ascending=False)).reset_index()

plt.figure(figsize=(12,12))

plt.subplot(211)
g = sns.countplot(x='sector', data=df_kiva)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g.set_xlabel("", fontsize=12)
g.set_ylabel("count", fontsize=12)
g.set_title("sector loan counts", fontsize=15)

plt.subplot(212)
g1 = sns.barplot(x='sector', y='loan_amount', data=sector_amount)
g1.set_xticklabels(g.get_xticklabels(), rotation=45)
g1.set_xlabel("sector", fontsize=12)
g1.set_ylabel("Avg loan amount", fontsize=12)
g1.set_title("Loan Amount Mean by sectors", fontsize=15)

plt.show()

df_kiva['loan_amount_log'] = np.log(df_kiva['loan_amount'])
df_kiva['funded_amount_log'] = np.log(df_kiva['funded_amount'])
df_kiva['diff_fund'] = df_kiva['loan_amount'] / df_kiva['funded_amount']

plt.figure(figsize=(12,14))

plt.subplot(312)
g1 = sns.boxplot(x='sector', y='loan_amount_log', data=df_kiva)
g1.set_xticklabels(g1.get_xticklabels(), rotation=45)
g1.set_xlabel("")
g2.set_ylabel("Loan Amount log", fontsize=12)
g2.set_title("Loan Dist by Sectors", fontsize=15)

plt.subplot(311)
g2 = sns.boxplot(x='sector', y='funded_amount_log', data=df_kiva)
g2.set_xticklabels(g2.get_xticklabels(), rotation=45)
g2.set_xlabel("")
g2.set_ylabel("Funded amount log", fontsize=12)
g2.set_title("Funded Dist by Sector", fontsize=15)

plt.subplot(313)
g3 = sns.boxplot(x='sector', y='term_in_months', data=df_kiva)
g3.set_xticklabels(g3.get_xticklabels(), rotation=45)
g3.set_xlabel("")
g3.set_ylabel("Term Months", fontsize=12)
g3.set_title("Term months by sector", fontsize=15)

plt.subplots_adjust(wspace=0.2, hspace=0.6, top=0.9)
plt.show()

activies = df_kiva.activity.value_counts()[:30]
activies_amount = pd.DataFrame(df_kiva.groupby(['activity'])['loan_amount'].mean().
                               sort_values(ascending=False)[:30]).reset_index()

plt.figure(figsize=(12,10))

plt.subplot(211)
g = sns.barplot(x=activies.index, y=activies.values)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_title("The 30 highest Fre Activities", fontsize=15)
g.set_xlabel("")
g.set_ylabel("count", fontsize=12)

plt.subplot(212)
g1 = sns.barplot(x='activity', y='loan_amount', data=activies_amount)
g1.set_xticklabels(g1.get_xticklabels(), rotation=90)
g1.set_xlabel("")
g1.set_ylabel("Avg Loan amount", fontsize=12)
g1.set_title("the 30 highest mean loan amount by activity", fontsize=15)

plt.subplots_adjust(wspace=0.2, hspace=0.8, top=0.9)
plt.show()

plt.figure(figsize=(6, 5))
g = sns.countplot(x='repayment_interval', data=df_kiva)
g.set_title("Repayment Interval Dist", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.show()

sns.FacetGrid(df_kiva, hue='repayment_interval', size=5, aspect=2).\
    map(sns.kdeplot, 'loan_amount_log', shade=True).add_legend()
plt.show()

df_kiva['lender_count_log'] = np.log(df_kiva['lender_count'] + 1)
sns.FacetGrid(df_kiva, hue='repayment_interval', size=5, aspect=2).\
    map(sns.kdeplot, 'lender_count_log', shade=True).add_legend()
plt.show()

sector_repay = ['sector', 'repayment_interval']
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(df_kiva[sector_repay[0]], df_kiva[sector_repay[1]]).style.background_gradient(cmap=cm)

df_kiva.loc[df_kiva.country == 'The Democratic Republic of the Congo', 'country'] = 'Republic of Congo'
df_kiva.loc[df_kiva.country == 'Saint Vincent and Grenadines', 'country'] = 'S Vinc e Grenadi'

country = df_kiva.country.value_counts()
country_amount = pd.DataFrame(
    df_kiva[df_kiva['loan_amount'] < 20000].
        groupby(['country'])['loan_amount'].mean().sort_values(ascending=False)[:35]).reset_index()

plt.figure(figsize=(10,14))
plt.subplot(311)
g = sns.barplot(x=country.index[:35], y=country.values[:35])
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel("")
g.set_ylabel("count", fontsize=12)
g.set_title("The 35 most freq helped countrys", fontsize=15)

plt.subplot(312)
g1 = sns.barplot(x=country_amount['country'], y=country_amount['loan_amount'])
g1.set_xticklabels(g.get_xticklabels(), rotation=90)
g1.set_xlabel("")
g1.set_ylabel("amount", fontsize=12)
g1.set_title("The 35 highest Mean's of Loan by Country", fontsize=15)

plt.subplot(313)
g2 = sns.countplot(x='world_region', data=df_kiva_loc)
g2.set_xticklabels(g2.get_xticklabels(), rotation=45)
g2.set_xlabel("")
g2.set_ylabel("Count", fontsize=12)
g2.set_title("World Regions Dist", fontsize=15)

plt.subplots_adjust(wspace = 0.2, hspace = 0.90,top = 0.9)
plt.show()

country_repayment = ['country', 'repayment_interval']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_kiva[country_repayment[0]], df_kiva[country_repayment[1]]).style.background_gradient(cmap=cm)

country_sector = ['country', 'sector']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_kiva[country_sector[0]], df_kiva[country_sector[1]]).style.background_gradient(cmap=cm)

currency = df_kiva['currency'].value_counts()

plt.figure(figsize=(6,5))
g = sns.barplot(x=currency.index[:35], y=currency.values[:35])
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel("")
g.set_ylabel("count", fontsize=12)
g.set_title("the 35 freq Currencies", fontsize=15)
plt.show()

df_kiva.borrower_genders = df_kiva.borrower_genders.astype(str)
df_sex = pd.DataFrame(df_kiva.borrower_genders.str.split(',').tolist())

df_kiva['sex_borrowers'] = df_sex[0]
df_kiva.loc[df_kiva.sex_borrowers == 'nan', 'sex_borrowers'] = np.nan

sex_mean = pd.DataFrame(df_kiva.groupby(['sex_borrowers'])['loan_amount'].mean().sort_values(ascending=False)).reset_index()

print("Gender Dist")
print(round(df_kiva['sex_borrowers'].value_counts() / len(df_kiva['sex_borrowers'])*100))

plt.figure(figsize=())
plt.subplot(321)
g = sns.countplot(x='sex_borrowers', data=df_kiva, order=['male', 'female'])
g.set_title("Gender Dist", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(322)
g1 = sns.barplot(x='sex_borrowers', y='loan_amount', data=sex_mean)
g1.set_title("mean loan by Gender", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("avg loan amount", fontsize=12)

plt.subplot(311)
g2 = sns.countplot(x='sector', data=df_kiva, hue='sex_borrowers', hue_order=['male', 'female'])
g2.set_xticklabels(g2.get_xticklabels(), rotation=45)
g2.set_title("exploring gender by sector", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("count", fontsize=12)
plt.show()

plt.figure(figsize=(10,6))

g = sns.countplot(x='sex_borrowers', data=df_kiva, hue='repayment_interval')
g.set_title("genders by repayment interval", fontsize=15)
g.set_xlabel("")
g.set_ylabel("count dist", fontsize=12)

plt.show()

df_kiva['date'] = pd.to_datetime(df_kiva['date'])
df_kiva['funded_time'] = pd.to_datetime(df_kiva['funded_time'])
df_kiva['posted_time'] = pd.to_datetime(df_kiva['posted_time'])

df_kiva['date_month_year'] = df_kiva['date'].dt.to_period("M")
df_kiva['funded_year'] = df_kiva['funded_time'].dt.to_period("M")
df_kiva['posted_month_year'] = df_kiva['posted_time'].dt.to_period("M")
df_kiva['date_year'] = df_kiva['date'].dt.to_period("A")
df_kiva['funded_year'] = df_kiva['funded_time'].dt.to_period("A")
df_kiva['posted_year'] = df_kiva['posted_time'].dt.to_period("A")

plt.figure(figsize=(10,14))

plt.subplot(311)
g = sns.countplot(x='date_month_year', data=df_kiva)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Month-year Loan Counting", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(312)
g1 = sns.pointplot(x='date_month_year', y='loan_amount', data=df_kiva, hue='repayment_interval')
g1.set_xticklabels(g1.get_xticklabels(), rotation=90)
g1.set_title("Mean loan by Month Year", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Loan Amount", fontsize=12)

plt.subplot(313)
g2 = sns.pointplot(x='date_month_year', y='term_in_months', data=df_kiva, hue='repayment_interval')
g2.set_xticklabels(g2.get_xticklabels(), rotation=90)
g2.set_title("term in months by months_year", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Term in Months", fontsize=12)

plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.9)
plt.show()

