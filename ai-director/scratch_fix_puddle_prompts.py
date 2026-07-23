import json, urllib.request

API = "http://127.0.0.1:8000"

# scene_number -> id
IDS = {
 1:"e07c78e3-b376-4d36-86a8-fb4b6084fa12",
 2:"9e9061ff-fb56-4430-8276-3cb90cb2235e",
 3:"ec659c33-adc9-485c-9abe-0d4b395b045e",
 4:"d041db33-5ec9-47c5-bb9d-4a4120b556eb",
 5:"ce0fa90d-4436-4aee-8f4b-015a61042751",
 6:"ba91caf5-4afe-47f4-bbee-c1ed3bc811f0",
 7:"9ac0c706-b767-4aaa-ba29-2180efe83d48",
 8:"de5c831a-aa21-46f5-a5c9-6fe1ecf5fc07",
 9:"06a5eadb-8d21-4f3e-830d-e9a6e282680d",
 10:"dbc7824a-0d95-4b15-a519-a0ee99318833",
 11:"f8600066-ed9f-4b1d-ba23-c097e2664e76",
 12:"e70a9d8f-9fe0-4e45-b227-dadd347ef890",
}

P = {
 1:("Wide aerial establishing shot craning slowly down over a sunny pastel toddler garden just after rain. "
    "Fat raindrops still drip from lemon-yellow flowers, a faint rainbow haze hangs in the clearing sky, and "
    "five small puddles are scattered across mint-green grass, each glowing a different colour: blue, green, "
    "gold, pink, purple. Warm golden morning sunbeams break through parting clouds. Soft 3D cartoon storybook "
    "render, glossy clean surfaces, gentle drifting mist, no characters yet, peaceful and bright."),
 2:("Low medium shot at grass level. A chubby baby elephant with soft dove-grey skin, a big round head, huge "
    "friendly dark eyes and a tiny curled trunk, wearing bright yellow rain boots, waddles happily out of a "
    "flowering archway into the garden; a small fluffy yellow duckling in a tiny red raincoat waddles right "
    "beside him. Dewy sparkles on the grass, a soft lens flare from the low sun. Plush toy-like textures, "
    "rounded shapes, cheerful pastel palette, gentle bouncing motion."),
 3:("Dynamic medium shot with a slight push-in. The dove-grey baby elephant in yellow boots stands at the rim "
    "of a bright BLUE puddle, one boot lifted mid-jump and trunk raised in delight. A big soft glowing number "
    "'1' made of bubbles floats above the blue water. Splashy droplets hang in the sunlight. Crisp 3D cartoon "
    "look, saturated friendly colour, motion-blurred water, pure joyful energy."),
 4:("Over-the-shoulder tracking shot from behind the elephant as he stomps both yellow boots into a round GREEN "
    "puddle, water fanning outward in a happy crown, wide grin and curled trunk flicking droplets. Two glossy "
    "bubbles shaped like the number '2' drift upward. Mint grass and lemon flowers frame the edges under dappled "
    "leaf-shadow light. Bouncy toddler-safe movement, plush rounded rendering, warm and playful."),
 5:("Hero low-angle slow-motion shot. The baby elephant leaps with both boots together into a shimmering puddle "
    "and a tall sparkling water fountain erupts around him like a crystal umbrella, the yellow duckling flapping "
    "gleefully in the spray. Bright midday sun scatters tiny rainbows through every droplet. Vivid pastel cartoon "
    "render, exaggerated joyful splash, a confetti of light, dove-grey hero clearly centred."),
 6:("Wide, gently bobbing shot. The dove-grey elephant and the red-coated duckling jump side by side, trunk and "
    "tiny wings thrown up in a cheering 'hooray', yellow boots kicking little splashes, the colourful puddles "
    "lined up behind them. Soft balloons of light drift past. Warm celebratory sunlight, glossy 3D storybook "
    "style, big open smiles, super-cute rounded proportions."),
 7:("Tender profile two-shot. The baby elephant kneels beside a glowing GOLD puddle while the yellow duckling in "
    "its red raincoat dips one webbed foot in, both leaning close to study their reflection. A honey-coloured "
    "number '3' shimmers on the water surface. Soft late-morning glow with floating dandelion fluff. Delicate "
    "plush cartoon look, calm friendly mood, gentle highlights on dove-grey skin."),
 8:("Medium close-up on a smooth side dolly. The elephant splashes through a soft PINK puddle, trunk curled and "
    "spraying a fine misty arc, the duckling scampering behind and leaving little ripples. Four heart-shaped "
    "bubbles carry a pastel number '4'. Drifting cherry-blossom petals and a warm backlit pink rim light. Clean "
    "rounded 3D render, sweet dreamy palette, lively but gentle motion."),
 9:("Dramatic low anticipation shot. The dove-grey baby elephant crouches at the brink of the biggest PURPLE "
    "puddle, yellow boots braced, cheeks puffed, huge eyes sparkling, while the duckling peeks out from under a "
    "wing. A glowing violet number '5' hovers overhead. Cool lavender shade meets warm sun in a held breath "
    "before the splash. Glossy plush cartoon style, playful suspense, storybook framing."),
 10:("Epic slow-motion wide shot. The elephant cannonballs into the purple puddle and a giant sparkling water "
     "crown bursts skyward, droplets scattering into all five puddle colours, the yellow duckling launched "
     "joyfully in the spray, both laughing out loud. Blazing cheerful sunbeams and a full droplet-rainbow. "
     "Ultra-cute 3D cartoon spectacle, maximal splash, radiant colour, pure toddler delight."),
 11:("Intimate low close-up. The baby elephant's bright yellow rain boots stand in shallow shimmering water, "
     "tiny concentric ripples spreading out, sun glinting off the wet rubber, the duckling's little red raincoat "
     "reflected beside them. Softly settling droplets, a calm winding-down feeling, warm amber golden-hour light. "
     "Delicate plush 3D render, cosy and content, quiet pastel tones."),
 12:("Closing crane-up wide shot. The dove-grey baby elephant and the yellow duckling stand together waving at "
     "the viewer in the middle of the sunny pastel garden, the five colourful puddles — blue, green, gold, pink, "
     "purple — lined up sparkling in a gentle arc behind them, soft bubbles rising into a clear sky. Storybook "
     "'the end' warmth, golden light, glossy rounded cartoon style, a happy farewell."),
}

for n, sid in IDS.items():
    body = json.dumps({"prompt": P[n]}).encode()
    req = urllib.request.Request(f"{API}/api/scenes/{sid}", data=body,
                                 method="PUT", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        print(f"scene {n:2d}: updated ({len(P[n])} chars)")
    except Exception as e:
        print(f"scene {n:2d}: FAILED {e}")
