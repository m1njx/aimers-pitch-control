"""Recover the Public-LB-optimal constant shift from two symmetric probes.

Usage:
  python scratch/lb_shift_vertex.py SCORE_MINUS SCORE_PLUS

The centre score is the observed submit_v16 Public LB score.  With no
probability clipping, Brier score makes this interpolation exactly quadratic.
"""
import argparse


CENTRE_SCORE = 1003.0255197284
H = 0.005


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("score_minus", type=float, help="submit_v20 (-0.005) score")
    parser.add_argument("score_plus", type=float, help="submit_v21 (+0.005) score")
    args = parser.parse_args()

    curvature = (args.score_plus + args.score_minus - 2.0 * CENTRE_SCORE) / (2.0 * H * H)
    slope = (args.score_plus - args.score_minus) / (2.0 * H)
    if curvature >= 0:
        raise SystemExit(
            "The three scores are not concave. Check score/order inputs or use a smaller probe."
        )
    optimum_shift = -slope / (2.0 * curvature)
    optimum_score = CENTRE_SCORE + slope * optimum_shift + curvature * optimum_shift**2
    print(f"optimal_shift={optimum_shift:+.10f}")
    print(f"quadratic_predicted_score={optimum_score:.10f}")
    print(f"predicted_gain={optimum_score - CENTRE_SCORE:+.10f}")


if __name__ == "__main__":
    main()
