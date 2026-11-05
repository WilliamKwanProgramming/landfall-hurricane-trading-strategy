# Landfall: Hurricane Event-Driven Trading Strategy

> **Status: work in progress.** This repository currently holds the project plan only. No code, results or performance numbers have been published yet.

## Question

Do equity and commodity prices fully reflect the information in National Hurricane Center (NHC) forecasts in the days around a hurricane landfall? This project builds a point-in-time dataset and tests whether trading the gap between forecast-implied exposure and realized price moves has a positive edge after costs.

## Planned approach

- **Point-in-time data.** Use forecasts exactly as issued (NHC ATCF a-decks), not the later best-track reanalysis, so no look-ahead enters the backtest. Best-track data (HURDAT2) is used only to measure forecast error.
- **Exposure scoring.** Score each storm's forecast track and intensity by expected wind exposure for listed insurers and reinsurers and for storm-sensitive commodities (orange juice, natural gas, lumber).
- **Event study and backtest.** Trade from 5 days before to 10 days after landfall, net of transaction costs. Validate leave-one-season-out, report block-bootstrap confidence intervals and deflated Sharpe ratios, and publish negative results.
- **Performance.** Footprint Monte Carlo and backtester in C++ (pybind11), checked against a NumPy reference implementation.

## Data

Public sources only: the NHC ATCF archive and HURDAT2, plus public price data. No vendor data is stored or redistributed here.

## Background

Roll (1984), "Orange Juice and Weather", *American Economic Review*.

## Disclaimer

Educational research project. Not investment advice.
