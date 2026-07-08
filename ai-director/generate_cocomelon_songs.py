"""
Generate 10 Cocomelon-style baby poem songs (4 min each)
ACE-Step 1.5 XL SFT — DualCLIPLoader, high steps for quality
"""
import sys, time, random, json, urllib.request, os
sys.path.insert(0, ".")

COMFYUI = "http://127.0.0.1:8188"
OUTPUT_DIR = r"D:\assets_genrated\ltx_output\cocomelon_songs"

SONGS = [
    {
        "name": "twinkle_star_lullaby",
        "tags": "children's nursery rhyme, cocomelon style, cheerful, bright, xylophone, ukulele, soft drums, claps, music box, playful, cute, baby song, animated show soundtrack",
        "lyrics": """[verse]
Twinkle twinkle little star
How I wonder what you are
Up above the world so high
Like a diamond in the sky

[chorus]
Shine shine shine so bright
Lighting up the dark dark night
Shine shine shine for me
Prettiest star that I can see

[verse]
When the blazing sun is gone
When he nothing shines upon
Then you show your little light
Twinkle twinkle all the night

[chorus]
Shine shine shine so bright
Lighting up the dark dark night
Shine shine shine for me
Prettiest star that I can see

[bridge]
La la la la la la la
Twinkle twinkle little star
La la la la la la la
I know just what you are

[outro]
Twinkle twinkle little star
How I wonder what you are""",
        "bpm": 100, "key": "C major",
    },
    {
        "name": "abc_fun_song",
        "tags": "children's nursery rhyme, cocomelon style, educational, fun, upbeat, piano, bells, tambourine, cheerful vocals, kids song, animated, bouncy, happy",
        "lyrics": """[verse]
A B C D E F G
Come and sing along with me
H I J K L M N
Let us sing it once again

[chorus]
A B C it's fun you see
Learning letters happily
One two three now sing with me
Easy as can be

[verse]
O P Q R S T U
I love learning how about you
V and W X Y Z
Now I know my ABCs

[chorus]
A B C it's fun you see
Learning letters happily
One two three now sing with me
Easy as can be

[bridge]
Clap your hands and stamp your feet
A B C is such a treat
Every letter has a sound
Letters letters all around

[chorus]
A B C it's fun you see
Learning letters happily
One two three now sing with me
Easy as can be""",
        "bpm": 110, "key": "G major",
    },
    {
        "name": "bath_time_splash",
        "tags": "children's music, cocomelon style, playful, bubbly, fun, marimba, kazoo, water sounds, giggly, toddler song, cute melody, bouncy rhythm, happy",
        "lyrics": """[verse]
Splish splash it's bath time now
Rubber ducky shows me how
Bubbles bubbles everywhere
Bubbles floating in the air

[chorus]
Bath time bath time splashy fun
Washing washing everyone
Scrub your toes and scrub your nose
Bath time fun before bed goes

[verse]
Soap and water nice and warm
Keeps me clean and keeps me warm
Wash my fingers wash my hair
Squeaky clean beyond compare

[chorus]
Bath time bath time splashy fun
Washing washing everyone
Scrub your toes and scrub your nose
Bath time fun before bed goes

[bridge]
Rubber ducky you're the one
Making bath time so much fun
Quack quack quack the ducky sings
Bath time joy is what it brings

[outro]
Splish splash bath time done
Now it's time for sleepy fun""",
        "bpm": 105, "key": "F major",
    },
    {
        "name": "wheels_on_bus",
        "tags": "children's nursery rhyme, cocomelon style, cheerful, driving rhythm, acoustic guitar, tambourine, hand claps, toddler music, sing along, animated soundtrack, bright",
        "lyrics": """[verse]
The wheels on the bus go round and round
Round and round round and round
The wheels on the bus go round and round
All through the town

[verse]
The wipers on the bus go swish swish swish
Swish swish swish swish swish swish
The wipers on the bus go swish swish swish
All through the town

[chorus]
Come ride the bus with me today
Singing songs along the way
Beep beep beep we're on our way
Having fun the whole day

[verse]
The horn on the bus goes beep beep beep
Beep beep beep beep beep beep
The horn on the bus goes beep beep beep
All through the town

[verse]
The babies on the bus go wah wah wah
Wah wah wah wah wah wah
The babies on the bus go wah wah wah
All through the town

[chorus]
Come ride the bus with me today
Singing songs along the way
Beep beep beep we're on our way
Having fun the whole day""",
        "bpm": 108, "key": "D major",
    },
    {
        "name": "animal_sounds_farm",
        "tags": "children's music, cocomelon style, farm animals, educational, fun, banjo, fiddle, acoustic guitar, country kids, playful, cheerful, toddler song, cute",
        "lyrics": """[verse]
Old MacDonald had a farm
E I E I O
And on his farm he had a cow
E I E I O

[chorus]
Moo moo here moo moo there
Here a moo there a moo
Everywhere a moo moo
Old MacDonald had a farm
E I E I O

[verse]
Old MacDonald had a farm
E I E I O
And on his farm he had a duck
E I E I O

[chorus]
Quack quack here quack quack there
Here a quack there a quack
Everywhere a quack quack
Old MacDonald had a farm
E I E I O

[bridge]
Oink oink baa baa neigh neigh neigh
Animals singing every day
Cluck cluck woof woof meow meow
Farm is fun I'll show you how

[verse]
Old MacDonald had a farm
E I E I O
And on his farm he had a sheep
E I E I O
Baa baa here baa baa there
Everywhere a baa baa""",
        "bpm": 112, "key": "A major",
    },
    {
        "name": "counting_123",
        "tags": "children's educational song, cocomelon style, counting, numbers, cheerful, glockenspiel, piano, light percussion, claps, sweet melody, toddler learning, animated, bright",
        "lyrics": """[verse]
One two three four five
Counting makes me feel alive
Six seven eight nine ten
Let us count them all again

[chorus]
One two three count with me
Four five six as easy as this
Seven eight nine we're doing fine
Ten ten ten let's count again

[verse]
How many fingers do you see
Count them now along with me
One two three four five
High five high five come alive

[chorus]
One two three count with me
Four five six as easy as this
Seven eight nine we're doing fine
Ten ten ten let's count again

[bridge]
Numbers numbers everywhere
Count the stars up in the air
Count your toes and count your nose
Counting counting here it goes

[outro]
One two three four five six seven
Eight nine ten we count to heaven
Learning numbers every day
Counting in a fun fun way""",
        "bpm": 100, "key": "Eb major",
    },
    {
        "name": "colors_rainbow",
        "tags": "children's music, cocomelon style, colorful, bright, happy, synthesizer, bells, xylophone, marimba, dreamy, educational, toddler song, magical, whimsical",
        "lyrics": """[verse]
Red and orange yellow too
Green and blue I love you
Purple pink and colors bright
Rainbow shining in the light

[chorus]
Colors colors everywhere
Colors colors in the air
Red blue yellow green
Prettiest colors ever seen

[verse]
Roses are red the sky is blue
Grass is green the sun shines through
Butterflies in colors fly
Painting pictures in the sky

[chorus]
Colors colors everywhere
Colors colors in the air
Red blue yellow green
Prettiest colors ever seen

[bridge]
Mix them up and what do you get
Orange purple something yet
Blue and yellow make it green
Most amazing thing I've seen

[chorus]
Colors colors everywhere
Colors colors in the air
Red blue yellow green
Prettiest colors ever seen""",
        "bpm": 95, "key": "Bb major",
    },
    {
        "name": "bedtime_sleepy",
        "tags": "children's lullaby, cocomelon style, gentle, soothing, music box, soft piano, strings, harp, calming, bedtime song, dreamy, warm, peaceful, tender",
        "lyrics": """[verse]
Close your eyes my little one
Today was full of so much fun
The moon is out the stars are bright
Time to say a sweet goodnight

[chorus]
Sleep sleep little baby sleep
Counting fluffy little sheep
One two three now close your eyes
Dream of rainbow butterflies

[verse]
Teddy bear is by your side
Blanket warm and snuggled tight
Mommy's here and daddy too
We will always love you true

[chorus]
Sleep sleep little baby sleep
Counting fluffy little sheep
One two three now close your eyes
Dream of rainbow butterflies

[bridge]
Hush now hush the world is still
Moonlight dancing on the hill
Stars are twinkling just for you
Sleep my darling through and through

[outro]
Goodnight moon and goodnight stars
Goodnight world both near and far
Sleep sleep sleep my little dear
Mommy daddy both are here""",
        "bpm": 72, "key": "Ab major",
    },
    {
        "name": "head_shoulders",
        "tags": "children's action song, cocomelon style, energetic, fun, body parts, educational, drums, claps, whistle, bouncy, toddler dance, upbeat, animated, happy",
        "lyrics": """[verse]
Head shoulders knees and toes
Knees and toes knees and toes
Head shoulders knees and toes
Eyes ears mouth and nose

[chorus]
Touch your head touch your toes
Wiggle wiggle here it goes
Clap your hands stomp your feet
Move your body to the beat

[verse]
Head shoulders knees and toes
Knees and toes knees and toes
Head shoulders knees and toes
Eyes ears mouth and nose

[bridge]
Faster faster here we go
Start out fast then go real slow
Reach up high and touch the sky
Bend down low then jump up high

[chorus]
Touch your head touch your toes
Wiggle wiggle here it goes
Clap your hands stomp your feet
Move your body to the beat

[verse]
Stretch your arms and turn around
Jump up high then touch the ground
Shake your hands and nod your head
Dance and dance until it's bed""",
        "bpm": 115, "key": "E major",
    },
    {
        "name": "itsy_bitsy_spider",
        "tags": "children's nursery rhyme, cocomelon style, gentle, storytelling, finger play, acoustic guitar, soft piano, light bells, sweet, toddler classic, animated, warm",
        "lyrics": """[verse]
The itsy bitsy spider
Climbed up the water spout
Down came the rain and
Washed the spider out

[chorus]
Out came the sun and
Dried up all the rain
And the itsy bitsy spider
Climbed up the spout again

[verse]
The itsy bitsy spider
Was climbing up so high
She wanted to reach the top
And touch the big blue sky

[chorus]
Out came the sun and
Dried up all the rain
And the itsy bitsy spider
Climbed up the spout again

[bridge]
Little spider don't give up
Keep on climbing to the top
Even when the rain comes down
You'll be the bravest one around

[verse]
The great big giant spider
Climbed up the water spout
Down came the rain and
Washed the spider out

[chorus]
Out came the sun and
Dried up all the rain
And the great big giant spider
Climbed up the spout again""",
        "bpm": 98, "key": "F major",
    },
]


def wait_comfyui(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"{COMFYUI}/system_stats", timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False

def free_vram():
    try:
        req = urllib.request.Request(f"{COMFYUI}/free", method="POST",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        time.sleep(2)
    except Exception:
        pass

def submit(workflow):
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(f"{COMFYUI}/prompt", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        return resp["prompt_id"], None
    except urllib.error.HTTPError as e:
        return None, e.read().decode()

def wait_completion(prompt_id, timeout=7200, poll=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = json.loads(urllib.request.urlopen(
                f"{COMFYUI}/history/{prompt_id}", timeout=10).read())
            if prompt_id in resp:
                return resp[prompt_id]
        except Exception:
            pass
        time.sleep(poll)
    raise TimeoutError(f"Timed out after {timeout}s")

def check_result(history):
    status = history.get("status", {})
    if status.get("status_str") == "error":
        msgs = status.get("messages", [])
        for msg in msgs:
            if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "execution_error":
                err = msg[1]
                return f"{err.get('node_type', '?')}: {err.get('exception_message', '?')[:300]}"
        return "Unknown error"
    return None

def collect_output(history, output_path):
    outputs = history.get("outputs", {})
    for node_id, node_out in outputs.items():
        if "audio" in node_out:
            for audio in node_out["audio"]:
                fname = audio["filename"]
                subfolder = audio.get("subfolder", "")
                url = f"{COMFYUI}/view?filename={fname}&subfolder={subfolder}&type=output"
                data = urllib.request.urlopen(url, timeout=120).read()
                with open(output_path, "wb") as f:
                    f.write(data)
                return len(data)
    return 0

def build_workflow(song, seed):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "acestep_v1.5_xl_sft_bf16.safetensors",
            "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "qwen_0.6b_ace15.safetensors",
            "clip_name2": "qwen_1.7b_ace15.safetensors",
            "type": "ace"}},
        "3": {"class_type": "VAELoader", "inputs": {
            "vae_name": "ace_1.5_vae.safetensors"}},
        "4": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
            "clip": ["2", 0],
            "tags": song["tags"],
            "lyrics": song["lyrics"],
            "seed": seed,
            "bpm": song["bpm"],
            "duration": 240.0,
            "timesignature": "4",
            "language": "en",
            "keyscale": song["key"],
            "generate_audio_codes": True,
            "cfg_scale": 2.5,
            "temperature": 0.8,
            "top_p": 0.92,
            "top_k": 0,
            "min_p": 0.0}},
        "5": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {
            "seconds": 240.0, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 0],
            "latent_image": ["5", 0], "seed": seed,
            "steps": 80,
            "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveAudio", "inputs": {
            "audio": ["7", 0], "filename_prefix": f"cocomelon_{song['name']}"}},
    }


def main():
    print("=" * 65)
    print("  COCOMELON-STYLE BABY POEMS — 10 Songs x 4 Minutes")
    print("  ACE-Step 1.5 XL SFT | 80 steps | DualCLIP | fp8")
    print("=" * 65)

    if not wait_comfyui(60):
        print("ComfyUI not reachable!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for i, song in enumerate(SONGS):
        seed = random.randint(0, 2**31 - 1)
        output_path = os.path.join(OUTPUT_DIR, f"{i+1:02d}_{song['name']}.wav")

        print(f"\n{'-'*65}")
        print(f"  [{i+1}/10] {song['name'].replace('_', ' ').title()}")
        print(f"  BPM: {song['bpm']} | Key: {song['key']} | Seed: {seed}")
        print(f"  Steps: 80 | Duration: 240s (4 min)")
        print(f"{'-'*65}")

        if i == 0:
            free_vram()

        wf = build_workflow(song, seed)
        prompt_id, err = submit(wf)
        if err:
            print(f"  REJECTED: {err[:200]}")
            results.append((song["name"], "REJECTED", 0))
            continue

        print(f"  Submitted: {prompt_id}")
        print(f"  Generating (LLM codes + 80 diffusion steps)...", flush=True)

        t0 = time.time()
        try:
            history = wait_completion(prompt_id, timeout=7200, poll=10.0)
        except TimeoutError:
            print(f"  TIMEOUT")
            results.append((song["name"], "TIMEOUT", 0))
            continue

        elapsed = time.time() - t0
        error = check_result(history)
        if error:
            print(f"  FAILED in {elapsed:.0f}s: {error}")
            results.append((song["name"], "FAILED", elapsed))
            continue

        size = collect_output(history, output_path)
        if size > 0:
            print(f"  OK — {elapsed:.0f}s ({elapsed/60:.1f}m) — {size/1024/1024:.1f} MB")
            results.append((song["name"], "OK", elapsed))
        else:
            print(f"  No audio output")
            results.append((song["name"], "NO_OUTPUT", elapsed))

    print(f"\n\n{'='*65}")
    print("  RESULTS SUMMARY")
    print(f"{'='*65}")
    ok = 0
    for name, status, elapsed in results:
        emoji = "+" if status == "OK" else "-"
        t = f"{elapsed:.0f}s" if elapsed else ""
        print(f"  [{emoji}] {name:30s} {status:12s} {t}")
        if status == "OK":
            ok += 1
    print(f"\n  {ok}/10 songs generated successfully")
    print(f"  Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
