"""Recover the optimal affine scale from submit_v22/v23 scores."""
import argparse


CENTER_SCALE = 1.0
CENTER_SCORE = 1011.2451265763
H = 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("score_scale_09", type=float, help="submit_v22 score")
    parser.add_argument("score_scale_11", type=float, help="submit_v23 score")
    args = parser.parse_args()

    curvature = (
        args.score_scale_11 + args.score_scale_09 - 2.0 * CENTER_SCORE
    ) / (2.0 * H * H)
    slope = (args.score_scale_11 - args.score_scale_09) / (2.0 * H)
    if curvature >= 0:
        raise SystemExit("Scores are not concave; check the inputs.")
    offset = -slope / (2.0 * curvature)
    optimum_scale = CENTER_SCALE + offset
    optimum_score = CENTER_SCORE + slope * offset + curvature * offset**2
    print(f"optimal_scale={optimum_scale:.10f}")
    print(f"quadratic_predicted_score={optimum_score:.10f}")
    print(f"predicted_gain_vs_v16={optimum_score - 1003.0255197284:+.10f}")


if __name__ == "__main__":
    main()
