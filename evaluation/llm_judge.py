
# data_mockup
# mock = [
#     {
#         image: "data/evaluation/images/img1.jpg",
#         gt_summary: "data/evaluation/gt/img1.txt",
#         image_type: "srh1",
#         baseline: "answer1"
#         case0: "answer2",
#         case3: "answer3"
#     }
# ]



# Sample data structure matching sample.json format
# Each entry in results list contains:
# - image: path to the meteorological image
# - cond_name: condition/parameter name (e.g., shr6, scp, tadv)
# - gt: ground truth weather summary
# - caption_gt: detailed explanation of the image for that condition
# - pretrained: output from pretrained model
# - frontier: output from frontier model
# - baseline: output from baseline model  
# - mtl: output from multi-task learning model

dataset_analysis_guide = {
    # ==========================================
    # 1. Kinematic Parameters (Wind & Shear)
    # ==========================================
    "shr6": {
        "Name": "0-6 km Bulk Wind Shear",
        "Analysis": (
            "Measure of wind speed/direction change from surface to 6km. "
            "Analyze values > 35-40 knots for supercell organization potential. "
            "High values indicate organized severe storms.\n\n"
            "[Interpretation] Look for values > 35-40 knots overlapping with instability. "
            "This threshold distinguishes organized supercells from ordinary pulse thunderstorms. "
            "Check if shear vectors cross storm motion vectors (perpendicular is better). "
            "[Critical Pitfall] Do not assume shear creates storms. Shear only organizes them. "
            "Without updraft initiation (CAPE/Lift), high shear is irrelevant."
        )
    },
    "srh1": {
        "Name": "0-1 km Storm Relative Helicity",
        "Analysis": (
            "Measure of low-level wind rotation potential relative to storm motion. "
            "Critical for tornado forecasting. Values > 100-200 m2/s2 suggest tornado potential.\n\n"
            "[Interpretation] Focus on values > 100-200 m2/s2, specifically near boundaries "
            "or warm fronts. Higher values indicate strong low-level rotation potential (tornadoes). "
            "[Critical Pitfall] High SRH is meaningless if the storm is elevated (not surface-based). "
            "Check for Convective Inhibition (CIN); if the surface air cannot rise, "
            "surface helicity will not be ingested."
        )
    },
    "stor": {
        "Name": "Storm Motion / Bunkers Storm Motion",
        "Analysis": (
            "Predicted motion vector of a supercell. "
            "Analyze to predict where the storm will move (Right-Mover vs Left-Mover). "
            "Essential for warning polygons and pathcasting.\n\n"
            "[Interpretation] Use the vector to visualize the 'Right-Mover' deviation. "
            "Essential for placing warning polygons downstream of the current storm location. "
            "[Critical Pitfall] These vectors assume supercell dynamics. Linear systems (QLCS) "
            "or disorganized storms will likely follow the mean wind, not the Bunkers vector."
        )
    },
    "effh": {
        "Name": "Effective-layer Storm-Relative Helicity (Eff. SRH)",
        "Analysis": (
            "Storm-relative helicity over the effective inflow layer. "
            "Large values with high CAPE favor rotating supercells and tornadoes.\n\n"
            "[Interpretation] More reliable than fixed-layer SRH. It calculates helicity "
            "only within the actual buoyant parcel layer. "
            "[Critical Pitfall] Be skeptical if the effective inflow layer is very thin or elevated. "
            "High numbers here don't guarantee tornado risk if the inflow base is far from the ground."
        )
    },
    
    # ==========================================
    # 2. Thermodynamic Parameters (Instability)
    # ==========================================
    "sbcp": {
        "Name": "Surface-Based CAPE (Convective Available Potential Energy)",
        "Analysis": (
            "Total energy available for a surface parcel to rise. "
            "Primary indicator of updraft strength. Values > 1000 J/kg support severe storms. "
            "Often visualized with CIN (Convective Inhibition) in composite maps.\n\n"
            "[Interpretation] Potential energy for updrafts. Look for gradients where CAPE "
            "is uncapped (low CIN). Extreme values (>3000 J/kg) support giant hail. "
            "[Critical Pitfall] CAPE is 'Potential' energy. It requires a trigger (lift). "
            "Also, beware of 'Ghost CAPE' caused by sensor errors (bad Dewpoint readings) "
            "creating unrealistic instability pockets."
        )
    },
    "lclh": {
        "Name": "Lifting Condensation Level (LCL) Height",
        "Analysis": (
            "Cloud base height. Lower values (< 1000m) correlate with higher tornado probability "
            "due to reduced evaporative cooling in downdrafts.\n\n"
            "[Interpretation] Cloud base height. Values < 750-1000m strongly correlate with "
            "significant tornadoes (reduced evaporative cooling in downdrafts). "
            "[Critical Pitfall] Extremely low LCLs may just mean stratus/fog (scud) with poor visibility. "
            "Conversely, high LCLs reduce tornado risk but significantly increase Microburst/Downburst wind risk."
        )
    },
    "laps": {
        "Name": "Mid-Level Lapse Rates (700-500 mb)",
        "Analysis": (
            "Rate of temperature decrease with height. Steeper rates (> 7 C/km) "
            "enhance instability (CAPE) and updraft acceleration (hail potential).\n\n"
            "[Interpretation] Steep rates (> 7 C/km) indicate cold air aloft, essential for "
            "rapid updraft acceleration and large hail growth. "
            "[Critical Pitfall] Steep mid-level rates are useless if low-level lapse rates "
            "are stable (inversion), preventing the parcel from ever reaching this layer."
        )
    },
    "lllr": {
        "Name": "Low-Level Lapse Rates (0-3 km)",
        "Analysis": (
            "Instability in the lowest 3km. Crucial for low-level stretching "
            "and non-supercell tornado potential. Steep lapse rates aid cold pool maintenance.\n\n"
            "[Interpretation] Values > 8 C/km suggest vigorous low-level mixing. "
            "Crucial for non-supercell tornadoes (Landspouts) and cold pool maintenance. "
            "[Critical Pitfall] Excessive lapse rates (> 9.5 C/km) often imply very dry air, "
            "which can mix out moisture and actually kill the storm updraft (Dry Adiabatic)."
        )
    },
    "fzlv": {
        "Name": "Freezing Level Height",
        "Analysis": (
            "Height where temperature is 0°C. Important for hail forecasting "
            "(lower freezing level = less melting) and flash flood efficiency (warm rain processes).\n\n"
            "[Interpretation] Used for hail and melting layer analysis. "
            "Optimal hail height is often balanced (not too high to melt, not too low to limit depth). "
            "[Critical Pitfall] Always adjust for terrain elevation. In high terrain, "
            "the freezing level is closer to the surface, drastically increasing hail/snow efficiency."
        )
    },
    "swbt": {
        "Name": "Surface Wet Bulb Temperature",
        "Analysis": (
            "Temperature a parcel would have if cooled to saturation. "
            "Used to discriminate rain/snow (freezing level estimation) and downdraft temperature.\n\n"
            "[Interpretation] Indicates the lowest temperature a parcel can reach via evaporation. "
            "Used to predict precipitation type (Rain vs Snow) and downdraft temperature potential."
        )
    },

    # ==========================================
    # 3. Composite Indices (Severe Weather)
    # ==========================================
    "scp": {
        "Name": "Supercell Composite Parameter",
        "Analysis": (
            "Combination of CAPE, Shear, and Helicity. "
            "Values > 1 indicate favorable environment for supercells. "
            "Normalize this input as a probability mask for supercell mode.\n\n"
            "[Interpretation] A composite index highlighting where CAPE, Shear, and Helicity overlap. "
            "Values > 1 suggest a parameter space favorable for supercells. "
            "[Critical Pitfall] This is a CONDITIONAL parameter. It does not predict storm initiation. "
            "A high SCP value on a clear, capped day means nothing if no storms form."
        )
    },
    "ttd": {
        "Name": "Surface Temperature / Dewpoint / MSLP Map",
        "Analysis": (
            "Surface T/Td and mean sea-level pressure analysis. "
            "Use gradients and convergence to locate fronts, drylines, and CI zones.\n\n"
            "[Interpretation] Use to identify surface boundaries (Fronts, Drylines). "
            "Look for tight gradients (packing) of isotherms or isodrosotherms. "
            "[Critical Pitfall] Automated maps often smooth out sharp boundaries. "
            "Always verify with actual wind barbs and station observations."
        )
    },
    "epvl": {
        "Name": "Equivalent Potential Vorticity (EPV)",
        "Analysis": (
            "EPV field to diagnose Conditional Symmetric Instability (CSI). "
            "Negative EPV with forcing/moisture favors slantwise convection and banded precipitation.\n\n"
            "[Interpretation] Negative EPV indicates Conditional Symmetric Instability (CSI). "
            "Look for this in winter storms or heavy rain events to predict banded precipitation. "
            "[Critical Pitfall] Negative EPV is not enough; it requires moisture and active forcing "
            "to realize the instability bands."
        )
    },
    
    # ==========================================
    # 4. Forcing & Moisture
    # ==========================================
    "tadv": {
        "Name": "Temperature Advection (usually 850mb)",
        "Analysis": (
            "Transport of warm/cold air by wind. Warm Air Advection (WAA) implies "
            "rising motion (forcing). Analyze for identifying frontal lifting zones.\n\n"
            "[Interpretation] Warm Air Advection (WAA) creates isentropic lift. "
            "Look for wind vectors crossing isotherms from warm to cold. "
            "[Critical Pitfall] Strong WAA can sometimes strengthen a "
            "Capping Inversion (warm nose aloft) rather than breaking it, depending on the layer."
        )
    },
    "thea": {
        "Name": "Theta-E Advection",
        "Analysis": (
            "Advection of Equivalent Potential Temperature. "
            "Identifies moisture/instability transport ridges. "
            "High Theta-E advection often precedes severe weather outbreaks.\n\n"
            "[Interpretation] Identifies the transport of unstable, moist air. "
            "Theta-E ridges often pinpoint the axis of highest instability before an outbreak."
        )
    },
    "mcon": {
        "Name": "Surface Moisture Convergence",
        "Analysis": (
            "Convergence of moisture flux. Primary short-term predictor for "
            "convective initiation (CI). Look for local maxima.\n\n"
            "[Interpretation] The primary short-term signal for Convective Initiation (CI). "
            "Look for persistent local maxima over time. "
            "[Critical Pitfall] This field is very noisy in models. "
            "Distinguish real convergence (boundaries) from model artifacts/terrain noise."
        )
    },
    "mcsm": {
        "Name": "MCS Maintenance Probability",
        "Analysis": (
            "Probability that existing MCSs will persist or strengthen. "
            "High values along/downstream of convection signal continued heavy rain and severe risk.\n\n"
            "[Interpretation] Probabilistic guidance for the longevity of storm complexes (MCS). "
            "Used to determine if a storm system will survive through the night or decay."
        )
    },
    "pchg": {
        "Name": "Surface Pressure Change (2-hour)",
        "Analysis": (
            "Isallobars. Rapid pressure falls indicate approaching lift/storms. "
            "Use to pinpoint frontal passage or storm approach.\n\n"
            "[Interpretation] Isallobars identify approaching lift. "
            "Rapid falls indicate the strongest forcing/approach of a storm system. "
            "[Critical Pitfall] Be aware of 'Atmospheric Tides' (Diurnal Variation). "
            "Pressure naturally drops in the afternoon; ensure the drop is synoptic, not just diurnal."
        )
    },

    # ==========================================
    # 5. Visual / Map Inputs (Crop Images)
    # ==========================================
    "rgnlrad_cropped": {
        "Name": "Regional Radar Reflectivity (Cropped)",
        "Analysis": (
            "Base reflectivity mosaic. Requires CNN/Vision encoder. "
            "Analyze for storm mode (Linear, Cellular, Bow Echo) and intensity (dBZ values).\n\n"
            "[Interpretation] Analyze storm morphology (Shape). "
            "Look for 'Kidney Beans' (Supercells), 'Bows' (Wind), or 'Lines' (QLCS). "
            "[Critical Pitfall] Not all red is rain. Watch out for 'Bright Banding' (melting snow), "
            "ground clutter (stationary), or biological echoes (birds/insects) which look different in texture."
        )
    },
    "bigsfc_cropped": {
        "Name": "Large Scale Surface Map (Cropped)",
        "Analysis": (
            "Synoptic surface observations (Fronts, T/Td, Wind barbs). "
            "Use Optical Character Recognition (OCR) or Vision model to identify "
            "frontal boundaries (Cold/Warm fronts) and pressure centers.\n\n"
            "[Interpretation] Visual analysis of pressure centers and frontal boundaries. "
            "Locate the 'Triple Point' (intersection of cold/warm/dry fronts) for high risk areas. "
            "[Critical Pitfall] OCR/Vision models may misread cluttered text. "
            "Rely on the geometry of isobars (troughs) rather than just text labels."
        )
    }
}



# LLM이 절대 비교는 못하더라도 상대 비교는 상대적으로 잘 하는 경향이 있음
# 임의의 수치적 비교는 금지 (ex: 10점 만점에 몇점)
# 평가 데이터셋 크기 약 5000개.

# 아이디어: 1. LLM Tier
# 1. gemini-3-pro-preview (max: 50 requests)
# 2. gemini-2.5-flash-preview-09-2025 or 2.5-pro (max: 500 requests)
# 3. gemini-2.5-flash-lite 

import os
import json
import time
from pathlib import Path
import random
from dataclasses import dataclass, asdict
from typing import List, Dict
from collections import Counter
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# .env 파일 로드
load_dotenv()

# ======================
# 설정
# ======================

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
TIER3_MODEL = "gemini-2.5-pro"
MODEL_NAMES = ["pretrained", "baseline", "mtl", "frontier"]

# ======================
# 데이터 구조
# ======================

@dataclass
class Sample:
    sample_id: str
    image_path: str
    cond_name: str
    image_type_desc: str
    gt: str
    caption_gt: str
    pretrained: str
    frontier: str
    baseline: str
    mtl: str

@dataclass
class JudgeResult:
    sample_id: str
    winner: str       # one of MODEL_NAMES or "tie"/"error"
    ranking: List[str]
    short_reason: str

# ======================
# Gemini 초기화 & 유틸
# ======================

SYSTEM_PROMPT_TIER3 = """
You are a lightweight automatic judge that compares four weather descriptions for the SAME meteorological image.

You receive (in this order):
- The meteorological image itself (first message content). This image shows United States meteorological data.
- A short guide about the image type and how to read it.
- Four anonymous candidate answers labeled A, B, C, D (labels are shuffled; model names are hidden).

Goal:
- Pick which single candidate best describes the actual content of the image, staying plausible for that image type.
- Also return a full ranking of all four candidates from best to worst (no duplicates).

Evaluation rules (image-only):
1) Fidelity to the image:
   - Prefer answers that clearly match visible/expected patterns for the given map/parameter type.
   - Penalize answers that invent contradictory or irrelevant phenomena not supported by the image.
2) Physical plausibility for the parameter:
   - The answer should make sense for the stated image type (e.g., shear vs. temperature advection).
3) Coverage and usefulness:
   - Prefer answers that capture key spatial patterns, gradients, or hazards implied by the image type, without generic filler.
4) Clarity:
   - If quality is similar, prefer the clearer, more coherent answer.

TIES:
- Use "tie" only if at least two answers are genuinely indistinguishable in quality.
- Otherwise, return a single winner. Ranking must still list all four labels exactly once (group any ties next to each other).

Output format (JSON only):
{
  "winner": "A" | "B" | "C" | "D" | "tie",
  "ranking": ["A", "B", "C", "D"],  // include all four labels once each, in order
  "short_reason": "One short sentence in English explaining the top choice."
}
""".strip()



def init_gemini():
    if not API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(
        model_name=TIER3_MODEL,
        system_instruction=SYSTEM_PROMPT_TIER3,
    )


def build_user_prompt(sample: Sample, masked_answers: List[Dict[str, str]]) -> str:
    answers_block = "\n\n".join(
        f"[ANSWER_{a['label']}]\n{a['answer']}" for a in masked_answers
    )
    return f"""
Use only the attached meteorological image when judging (no ground truth provided). Image path: {sample.image_path}

You will see four anonymous answers labeled A, B, C, and D (labels are shuffled).
Pick the best answer, provide a ranking of the labels, and reply with JSON only.

[IMAGE_TYPE_EXPLANATION]
{sample.image_type_desc}

{answers_block}
""".strip()


def extract_json(text: str) -> str:
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1:
        raise ValueError(f"JSON not found in response: {text}")
    return text[first:last+1]


def resolve_image_path(image_path: str) -> str:
    """Return an absolute path for the image, handling both absolute and relative inputs."""
    path = Path(image_path).expanduser()
    if path.is_absolute():
        return str(path)
    return str(Path.cwd() / path)


# ======================
# Tier3 단일 샘플 평가 (4모델 동시)
# ======================

def judge_sample_tier3(
    model,
    sample: Sample,
    max_retry: int = 3,
    retry_sleep: int = 5,
) -> JudgeResult:
    answer_map = {
        "pretrained": sample.pretrained,
        "baseline": sample.baseline,
        "mtl": sample.mtl,
        "frontier": sample.frontier,
    }

    # Shuffle model->label mapping so API does not see model names
    items = list(answer_map.items())
    random.shuffle(items)
    labels = ["A", "B", "C", "D"]
    masked_answers = []
    label_to_model = {}
    for label, (model_name, answer_text) in zip(labels, items):
        masked_answers.append({"label": label, "answer": answer_text})
        label_to_model[label] = model_name

    prompt = build_user_prompt(sample, masked_answers)
    image_path = resolve_image_path(sample.image_path)

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[Tier3] failed to load image for sample={sample.sample_id}: {e}")
        return JudgeResult(
            sample_id=sample.sample_id,
            winner="error",
            ranking=[],
            short_reason=f"Image load failed: {e}",
        )

    # API 요청 전 쿼리 정보 출력 (매핑은 콘솔에만 표시, 프롬프트에는 노출하지 않음)
    print(f"\n{'='*80}")
    print(f"[API REQUEST] sample_id={sample.sample_id}")
    print(f"Image: {image_path}")
    print(f"Condition: {sample.cond_name}")
    print(f"Label mapping (hidden from API): {label_to_model}")
    print(f"Prompt (first 500 chars):\n{prompt[:500]}...")
    print(f"{'='*80}\n")

    for attempt in range(max_retry):
        try:
            resp = model.generate_content(
                contents=[img, prompt],
                generation_config={"temperature": 0.0},
            )
            text = resp.text
            data = json.loads(extract_json(text))

            winner_label = str(data.get("winner", "error")).strip()
            ranking_labels = data.get("ranking", [])
            reason = data.get("short_reason", "")

            # Map back to real model names
            if winner_label.upper() in label_to_model:
                winner = label_to_model[winner_label.upper()]
            else:
                winner = winner_label

            ranking = []
            for lbl in ranking_labels:
                lbl_norm = str(lbl).strip().upper()
                ranking.append(label_to_model.get(lbl_norm, lbl_norm))
            
            # API 응답 출력
            print(f"\n{'='*80}")
            print(f"[API RESPONSE] sample_id={sample.sample_id}")
            print(f"Winner label -> model: {winner_label} -> {winner}")
            print(f"Ranking labels -> models: {ranking_labels} -> {ranking}")
            print(f"Reason: {reason}")
            print(f"Full response: {text}")
            print(f"{'='*80}\n")
            
            return JudgeResult(
                sample_id=sample.sample_id,
                winner=winner,
                ranking=ranking,
                short_reason=reason,
            )
        except Exception as e:
            print(f"[Tier3] error sample={sample.sample_id}, attempt={attempt+1}: {e}")
            time.sleep(retry_sleep)

    return JudgeResult(
        sample_id=sample.sample_id,
        winner="error",
        ranking=[],
        short_reason="Tier3 call failed",
    )


# ======================
# Tier3 전체 실행 (5000개)
# ======================

def run_tier3(samples: List[Sample], output_path: str, exclude_ids: Optional[set] = None):
    """
    각 샘플에 대해 4개 모델(pretrained, baseline, mtl, frontier) 답변을 한 번에 보내
    가장 좋은 답변과 전체 순위를 JSONL로 저장한다. exclude_ids에 있는 sample_id는 건너뜀.
    """
    if exclude_ids is None:
        exclude_ids = set()
    
    model = init_gemini()

    total_samples = len(samples)
    skipped_count = 0
    processed_count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            if s.sample_id in exclude_ids:
                skipped_count += 1
                print(f"[Tier3] {i+1}/{total_samples} sample_id={s.sample_id} - SKIPPED (in exclude list)")
                continue
            
            processed_count += 1
            print(f"[Tier3] {i+1}/{total_samples} (processed: {processed_count}, skipped: {skipped_count}) sample_id={s.sample_id}")
            result = judge_sample_tier3(model, s)
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            f.flush()
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Total samples: {total_samples}")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")


# ======================
# Conflict / Transitivity 판별
# ======================

def analyze_results(results_path: str, exclude_json_path: Optional[str] = None):
    """
    단일 샘플 평가(JSONL)를 읽어서 승자 분포, 오류/순위 이상 케이스를 요약한다.
    exclude_json_path가 주어지면 정상적으로 평가된 sample_id를 exclude 리스트에 추가한다.
    """
    winners = Counter()
    error_samples = []
    invalid_ranking_samples = []
    processed_ids = set()
    good_samples = set()

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            sample_id = d.get("sample_id")
            processed_ids.add(sample_id)
            winner = d.get("winner", "error")
            ranking = d.get("ranking", [])

            if winner == "error":
                error_samples.append(sample_id)
                continue

            winners[winner] += 1

            ranking_set = set(map(str, ranking))
            if len(ranking) != len(MODEL_NAMES) or ranking_set != set(MODEL_NAMES):
                invalid_ranking_samples.append(sample_id)
                continue

            good_samples.add(sample_id)

    print("=== 분석 결과 ===")
    print(f"총 결과 수        : {len(processed_ids)}")
    print(f"오류 샘플 수      : {len(error_samples)}")
    print(f"순위 이상 샘플 수 : {len(invalid_ranking_samples)}")
    print("승자 분포:")
    for name, count in winners.most_common():
        print(f"  - {name}: {count}")
    
    # 정상 샘플들을 exclude_json에 추가
    if exclude_json_path:
        # 기존 exclude list 로드
        existing_excludes = set()
        if os.path.exists(exclude_json_path):
            try:
                with open(exclude_json_path, 'r', encoding='utf-8') as f:
                    existing_excludes = set(json.load(f))
                print(f"\nLoaded {len(existing_excludes)} existing excluded IDs from {exclude_json_path}")
            except Exception as e:
                print(f"\nWarning: Could not load existing exclude list: {e}")
        
        # 기존 리스트와 병합
        updated_excludes = existing_excludes.union(good_samples)
        newly_added = updated_excludes - existing_excludes
        
        # JSON 파일에 저장 (정렬된 리스트로)
        with open(exclude_json_path, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(updated_excludes)), f, indent=4, ensure_ascii=False)
        
        print(f"\nUpdated {exclude_json_path}:")
        print(f"  - Previously excluded: {len(existing_excludes)}")
        print(f"  - Newly added: {len(newly_added)}")
        print(f"  - Total excluded: {len(updated_excludes)}")
        print(f"  - Good samples added : {len(good_samples)}")

    # 필요하면 리스트도 리턴
    return {
        "error_samples": error_samples,
        "invalid_ranking_samples": invalid_ranking_samples,
    }


# ======================
# 사용 예시 (뼈대)
# ======================

def load_samples_from_json(json_path: str) -> List[Sample]:
    """
    sample.json 형식의 파일을 읽어서 Sample 리스트 반환.
    JSON 구조:
    {
      "emb_scores": {...},
      "results": [
        {
          "image": "path/to/image",
          "cond_name": "shr6",
          "gt": "ground truth text",
          "caption_gt": "detailed caption",
          "pretrained": "model output",
          "frontier": "model output",
          "baseline": "model output",
          "mtl": "model output"
        },
        ...
      ]
    }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    results = data.get('results', [])
    
    for idx, item in enumerate(results):
        cond_name = item.get('cond_name', '')
        image_path = item.get('image', '')
        
        # Extract filename without extension as sample_id
        # e.g., "data/images/md0398_20180513_17_ttd.png" -> "md0398_20180513_17_ttd"
        filename = os.path.basename(image_path)
        sample_id = os.path.splitext(filename)[0]
        
        # Get image type description from dataset_analysis_guide
        image_type_desc = ""
        if cond_name in dataset_analysis_guide:
            guide = dataset_analysis_guide[cond_name]
            name = guide.get('Name', '')
            analysis = guide.get('Analysis', '')
            image_type_desc = f"{name}\n\n{analysis}"
        
        sample = Sample(
            sample_id=sample_id,
            image_path=image_path,
            cond_name=cond_name,
            image_type_desc=image_type_desc,
            gt=item.get('gt', ''),
            caption_gt=item.get('caption_gt', ''),
            pretrained=item.get('pretrained', ''),
            frontier=item.get('frontier', ''),
            baseline=item.get('baseline', ''),
            mtl=item.get('mtl', '')
        )
        samples.append(sample)
    
    return samples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', type=str, required=True,
                       help='Path to input JSON file (sample.json format)')
    parser.add_argument('--output-jsonl', type=str, default='tier3_results.jsonl',
                       help='Path to output JSONL file')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'analyze'], default='evaluate',
                       help='Mode: evaluate (run LLM judge) or analyze (analyze existing results)')
    parser.add_argument('--exclude-json', type=str, default='sample_exclude_id.json',
                       help='Path to JSON file containing sample IDs to exclude')
    parser.add_argument('--all-eval', action='store_true',
                       help='Evaluate all samples, ignoring exclude list')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Evaluate only the first N samples from the input JSON')
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        # 1) 샘플 로드
        print(f"Loading samples from {args.input_json}...")
        samples = load_samples_from_json(args.input_json)
        print(f"Loaded {len(samples)} samples")

        # 1-1) 부분 실행 옵션 처리
        if args.max_samples is not None and args.max_samples >= 0:
            original_count = len(samples)
            samples = samples[:args.max_samples]
            print(f"max-samples set to {args.max_samples}: evaluating first {len(samples)} of {original_count} samples")
        
        # 2) Exclude list 로드 (--all-eval이 아닐 때만)
        exclude_ids = set()
        if not args.all_eval:
            if os.path.exists(args.exclude_json):
                try:
                    with open(args.exclude_json, 'r', encoding='utf-8') as f:
                        exclude_ids = set(json.load(f))
                    print(f"Loaded {len(exclude_ids)} excluded IDs from {args.exclude_json}")
                except Exception as e:
                    print(f"Warning: Could not load exclude list: {e}")
            else:
                print(f"Exclude list file not found: {args.exclude_json}")
        else:
            print("--all-eval flag set: evaluating all samples")
        
        # 3) Tier3 평가
        print(f"Running Tier3 evaluation...")
        run_tier3(samples, args.output_jsonl, exclude_ids)
        print(f"Results saved to {args.output_jsonl}")
    
    elif args.mode == 'analyze':
        # 4) 결과 분석
        print(f"Analyzing results from {args.output_jsonl}...")
        analyze_results(args.output_jsonl, args.exclude_json)
