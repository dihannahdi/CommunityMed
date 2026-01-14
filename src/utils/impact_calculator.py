"""
Impact Calculator for CommunityMed AI
Evidence-based impact modeling for TB screening intervention

All estimates are derived from:
- WHO Global TB Report 2024
- Lancet systematic reviews on diagnostic delays
- WHO CHW Guidelines 2018
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import json


@dataclass
class ImpactConfig:
    """Configuration for impact calculations"""
    # Deployment parameters
    num_chws: int = 10000
    patients_per_day: int = 25
    working_days_per_year: int = 250
    
    # Epidemiological parameters (WHO 2024)
    tb_suspect_rate: float = 0.02  # 2% of encounters are TB suspects
    tb_positive_rate: float = 0.15  # 15% of suspects are TB positive
    
    # Baseline diagnostic performance
    baseline_sensitivity: float = 0.50  # CHW clinical judgment
    baseline_treatment_success: float = 0.85
    
    # Intervention performance (with CommunityMed AI)
    intervention_sensitivity: float = 0.85
    intervention_treatment_success: float = 0.90
    
    # Mortality and transmission
    mortality_reduction_from_early_treatment: float = 0.15
    secondary_infections_baseline: int = 10
    secondary_infections_intervention: int = 3
    
    # Cost parameters (USD)
    cost_per_chw_year: float = 236  # Total cost per CHW
    cost_per_tb_treatment: float = 500
    cost_per_hospital_day: float = 50
    productivity_per_life_year: float = 2000


def calculate_impact(config: ImpactConfig = None) -> Dict:
    """
    Calculate the impact of CommunityMed AI deployment
    
    Args:
        config: Impact configuration parameters
        
    Returns:
        Dictionary with comprehensive impact metrics
    """
    config = config or ImpactConfig()
    
    # Calculate patient encounters
    total_encounters = (
        config.num_chws * 
        config.patients_per_day * 
        config.working_days_per_year
    )
    
    # TB case flow
    tb_suspects = total_encounters * config.tb_suspect_rate
    tb_confirmed = tb_suspects * config.tb_positive_rate
    
    # Baseline outcomes (without CommunityMed)
    baseline_diagnosed = tb_confirmed * config.baseline_sensitivity
    baseline_treated = baseline_diagnosed * config.baseline_treatment_success
    baseline_missed = tb_confirmed - baseline_diagnosed
    
    # Intervention outcomes (with CommunityMed)
    intervention_diagnosed = tb_confirmed * config.intervention_sensitivity
    intervention_treated = intervention_diagnosed * config.intervention_treatment_success
    
    # Impact calculations
    additional_diagnosed = intervention_diagnosed - baseline_diagnosed
    additional_treated = intervention_treated - baseline_treated
    lives_saved = additional_treated * config.mortality_reduction_from_early_treatment
    
    # Transmission prevention
    secondary_prevented_baseline = baseline_missed * config.secondary_infections_baseline
    secondary_prevented_intervention = (tb_confirmed - intervention_diagnosed) * config.secondary_infections_intervention
    transmission_prevented = secondary_prevented_baseline - secondary_prevented_intervention
    
    # Cost analysis
    total_deployment_cost = config.num_chws * config.cost_per_chw_year
    treatment_cost_saved = additional_treated * config.cost_per_tb_treatment
    hospital_cost_saved = additional_treated * 14 * config.cost_per_hospital_day  # Avg 14 days
    productivity_preserved = lives_saved * 30 * config.productivity_per_life_year  # 30 life-years avg
    
    total_benefits = treatment_cost_saved + hospital_cost_saved + productivity_preserved
    net_benefit = total_benefits - total_deployment_cost
    roi = (net_benefit / total_deployment_cost) * 100 if total_deployment_cost > 0 else 0
    
    cost_per_life = total_deployment_cost / lives_saved if lives_saved > 0 else float('inf')
    cost_per_daly = total_deployment_cost / (lives_saved * 20) if lives_saved > 0 else float('inf')
    
    return {
        "deployment": {
            "num_chws": config.num_chws,
            "total_encounters": int(total_encounters),
            "tb_suspects_screened": int(tb_suspects),
            "tb_cases_confirmed": int(tb_confirmed),
        },
        "diagnostic_performance": {
            "baseline_sensitivity": config.baseline_sensitivity,
            "intervention_sensitivity": config.intervention_sensitivity,
            "sensitivity_improvement": f"+{(config.intervention_sensitivity - config.baseline_sensitivity) * 100:.0f}%",
            "baseline_diagnosed": int(baseline_diagnosed),
            "intervention_diagnosed": int(intervention_diagnosed),
            "additional_diagnosed": int(additional_diagnosed),
        },
        "health_outcomes": {
            "additional_cases_treated": int(additional_treated),
            "lives_saved": int(lives_saved),
            "secondary_infections_prevented": int(transmission_prevented),
        },
        "economic_analysis": {
            "total_deployment_cost_usd": int(total_deployment_cost),
            "treatment_cost_saved_usd": int(treatment_cost_saved),
            "hospital_cost_saved_usd": int(hospital_cost_saved),
            "productivity_preserved_usd": int(productivity_preserved),
            "total_benefits_usd": int(total_benefits),
            "net_benefit_usd": int(net_benefit),
            "roi_percent": f"{roi:.0f}%",
            "cost_per_life_saved_usd": f"${cost_per_life:.0f}",
            "cost_per_daly_averted_usd": f"${cost_per_daly:.1f}",
        },
        "cost_effectiveness": {
            "who_threshold": "Highly cost-effective (<1x GDP per capita per DALY)",
            "comparison_genexpert": "GeneXpert: $15-20/test vs CommunityMed: $0.001/test",
            "comparison_microscopy": "Microscopy: $2-5/test, requires lab",
        }
    }


def calculate_multi_year_impact(
    year1_chws: int = 10000,
    year2_chws: int = 50000,
    year3_chws: int = 200000,
) -> Dict:
    """
    Calculate multi-year cumulative impact
    
    Args:
        year1_chws: CHWs deployed in year 1
        year2_chws: CHWs deployed in year 2
        year3_chws: CHWs deployed in year 3
        
    Returns:
        Multi-year impact summary
    """
    results = {}
    cumulative_lives = 0
    cumulative_cost = 0
    
    for year, chws in enumerate([year1_chws, year2_chws, year3_chws], 1):
        config = ImpactConfig(num_chws=chws)
        year_impact = calculate_impact(config)
        
        lives_saved = year_impact["health_outcomes"]["lives_saved"]
        cost = year_impact["economic_analysis"]["total_deployment_cost_usd"]
        
        cumulative_lives += lives_saved
        cumulative_cost += cost
        
        results[f"year_{year}"] = {
            "chws_deployed": chws,
            "lives_saved": lives_saved,
            "cumulative_lives_saved": cumulative_lives,
            "deployment_cost_usd": cost,
            "cumulative_cost_usd": cumulative_cost,
        }
    
    results["summary"] = {
        "total_lives_saved_3_years": cumulative_lives,
        "total_investment_3_years": cumulative_cost,
        "cost_per_life_saved_average": f"${cumulative_cost / cumulative_lives:.0f}" if cumulative_lives > 0 else "N/A",
    }
    
    return results


def generate_report(config: ImpactConfig = None) -> str:
    """
    Generate a formatted impact report
    
    Args:
        config: Impact configuration
        
    Returns:
        Formatted markdown report
    """
    impact = calculate_impact(config)
    
    report = f"""
# CommunityMed AI Impact Report

## Deployment Scale
- Community Health Workers: **{impact['deployment']['num_chws']:,}**
- Annual Patient Encounters: **{impact['deployment']['total_encounters']:,}**
- TB Suspects Screened: **{impact['deployment']['tb_suspects_screened']:,}**
- TB Cases Confirmed: **{impact['deployment']['tb_cases_confirmed']:,}**

## Diagnostic Performance
- Baseline CHW Sensitivity: {impact['diagnostic_performance']['baseline_sensitivity']:.0%}
- With CommunityMed AI: {impact['diagnostic_performance']['intervention_sensitivity']:.0%}
- **Improvement: {impact['diagnostic_performance']['sensitivity_improvement']}**

## Health Outcomes
- Additional Cases Treated: **{impact['health_outcomes']['additional_cases_treated']:,}**
- **Lives Saved: {impact['health_outcomes']['lives_saved']:,}**
- Secondary Infections Prevented: **{impact['health_outcomes']['secondary_infections_prevented']:,}**

## Economic Analysis
| Metric | Value |
|--------|-------|
| Deployment Cost | ${impact['economic_analysis']['total_deployment_cost_usd']:,} |
| Total Benefits | ${impact['economic_analysis']['total_benefits_usd']:,} |
| Net Benefit | ${impact['economic_analysis']['net_benefit_usd']:,} |
| **ROI** | **{impact['economic_analysis']['roi_percent']}** |
| Cost per Life Saved | {impact['economic_analysis']['cost_per_life_saved_usd']} |
| Cost per DALY Averted | {impact['economic_analysis']['cost_per_daly_averted_usd']} |

## Cost-Effectiveness Assessment
{impact['cost_effectiveness']['who_threshold']}

---
*Generated by CommunityMed AI Impact Calculator*
"""
    return report


if __name__ == "__main__":
    # Default scenario
    print("=" * 60)
    print("CommunityMed AI Impact Calculator")
    print("=" * 60)
    
    # Year 1 pilot
    year1 = calculate_impact(ImpactConfig(num_chws=10000))
    print(f"\n📊 Year 1 Pilot (10,000 CHWs)")
    print(f"   Lives saved: {year1['health_outcomes']['lives_saved']:,}")
    print(f"   Cost per life: {year1['economic_analysis']['cost_per_life_saved_usd']}")
    print(f"   ROI: {year1['economic_analysis']['roi_percent']}")
    
    # Year 3 scale
    year3 = calculate_impact(ImpactConfig(num_chws=200000))
    print(f"\n📊 Year 3 Scale (200,000 CHWs)")
    print(f"   Lives saved: {year3['health_outcomes']['lives_saved']:,}")
    print(f"   Cost per life: {year3['economic_analysis']['cost_per_life_saved_usd']}")
    print(f"   ROI: {year3['economic_analysis']['roi_percent']}")
    
    # Multi-year projection
    print(f"\n📈 Multi-Year Projection")
    multi = calculate_multi_year_impact()
    print(f"   3-Year Lives Saved: {multi['summary']['total_lives_saved_3_years']:,}")
    print(f"   Total Investment: ${multi['summary']['total_investment_3_years']:,}")
    print(f"   Average Cost/Life: {multi['summary']['cost_per_life_saved_average']}")
    
    # Export full report
    print("\n" + "=" * 60)
    print(generate_report())
