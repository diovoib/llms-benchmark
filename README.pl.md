**Język:** Polski · [English](README.md)

GitHub i GitLab zawsze pokazują domyślnie `README.md`. Kliknij powyżej aby wybrać inny język.

# Lokalny benchmark agenta (narzędzia + pętla + kod)

Ten katalog służy do oceny **małego lokalnego modelu LLM** jako silnika agenta: główny nacisk kładziony tutaj jest na kwestię czy poprawnie woła udostępnione mu przez harness narzędzia, czy umie zorganizować sobie pętlę **tool** → **wynik** → **kolejny krok**, i czy w dłuższym zadaniu kodowania potrafi sprawdzić swoją pracę i robi faktyczny postęp oraz kiedy uznaje że zadanie jest skończone.

Serwer modelu jest poza benchmarkiem, uruchamiany niezależnie. Może być dowolny serwer (llama-server, Ollama, LM Studio, vLLM) wystawiający OpenAI API oraz obsługujący możliwość uruchamiania narzędzi w sposób wymagany przez model. Benchmark woła `POST {base_url}/chat/completions` przez oficjalne OpenAI SDK (`tools` / `tool_calls` / `role: tool`).

Efekt finalny powinien być oceniony przez sędziego, człowieka albo większy LLM w celu weryfikacji "Czy agent finalnie zrealizował zlecone zadanie". Czasem produkują opisy tekstowe i zadanie jest zrealizowane, ale niekoniecznie pasowałoby do regexpa. A czasem produkują za dużo, odpowiedź niby pasuje do regexpa ale jest to pytanie, albo zawiera śmieci jak np kawałki szablonu XML, json czy innego jakiego model używa.


## Wymagania

- Działający serwer OpenAI-compatible (llama: `http://127.0.0.1:8080/v1`, ollama: `http://127.0.0.1:11434/v1`, etc)
- Python 3.10+
- Python requirements:
  - openai>=1.40.0
  - pyyaml>=6.0
  - pytest>=8.0
  - httpx>=0.27.0
- Szablon czatu modelu **musi obsługiwać narzędzia**. (Np. Phi-4 nie obsługuje ich w poprawny sposób a phi-4-mini już tak)



## Setup

1. Ściągnij serwer, który hostuje modele (GGUF albo inne) i wystawia OpenAI-compatible `POST /v1/chat/completions`. Aktualnie wspierane są llama.cpp i ollama. Ale LM Studio i vLLM również będą działać, gdy `base_url` i klucz API się zgadzają między uruchomionym serwerem hostującym modele i plikiem konfiguracyjnym.

2.1. Dla llama.cpp skopiuj `llama.bat.example` do `llama.bat` (albo `llama.sh.example` do `llama.sh`) i wyedytuj tę kopię: uzupełnij ścieżkę do `llama-server`, katalog modeli `--models-dir`, `--ctx-size`, `--api-key`. W configu yaml ustaw llama.bat. Ściągnij modele, np z https://huggingface.co/models, szukaj wersji gguf.

```text
copy llama.bat.example llama.bat
```

Na przykład:

```text
"<llama-server.exe>" --models-dir "<katalog_wag>" --models-max 1 --ctx-size 16384 --parallel 1 --threads 8 -lv 3 --jinja --api-key "<api_key>"
```

Lub dla ollama

2.2. Dla ollama skopiuj `ollama.bat.example` do `ollama.bat` (albo `ollama.sh.example` do `ollama.sh`) i wyedytuj tę kopię: uzupełnij ścieżkę do katalogu modeli. W yaml ustaw `launcher: ../ollama.bat` (albo `../ollama.sh`). Modele ściąga się osobno (`ollama pull …`) i wpisujesz ich nazwy do configu yaml.

W yaml jest pole `launcher` od ustawienia, który rodzaj serwera będzie uruchamiany oraz skąd czytać klucz API i context size.

3. Python 3.10+ oraz zainstaluj zależności dla pythona. Z katalogu `bench/`:

```text
python -m pip install -r requirements.txt
```

4. Bedąc w katalogu `bench/` Uruchom serwer (`llama.bat`/`ollama.bat`) i zostaw go włączonego. Zerknij na początkowe logi serwera. Stąd możesz wziąć nazwy modeli, które serwer znalazł i wpisać te, których chcesz użyć, do pliku konfiguracyjnego (pkt 5). Zobacz czy w ogóle je znalazł, popraw błedy jeśli jakieś są.

llama-serwer uruchamia lokalnie Web UI z chatem - możesz sprawdzić czy działa otwierając w przeglądarce: `http://127.0.0.1:8080/`.

Niezależnie od serwera powinno działać coś takiego (z dokładnością do portu - port domyślny różni się między llama, olama i innymi):

```text
curl -H "Authorization: Bearer <api_key>" http://127.0.0.1:8080/v1/models
```

5. Skopiuj i wyedytuj `[bench/config.yaml.example](bench/config.yaml.example)` do `config.yaml`
- `base_url` — np. `http://127.0.0.1:8080/v1`
- `models[0].name` — **dokładna nazwa modelu z routera**
- `models[0].recommended_temperature` — używana przez profil `real` (`temperature: null`, jeśli ma być domyślna modelu)
- `prompt_variants` — jakich profili użyć gdy nie podane w opcjach startowych bencha, patrz niżej.

6. Po wejściu w command line do katalogu `bench/` uruchom poniższą komendę:

```text
python run.py run
```

7. Po uruchomieniu bench tworzy `bench/results/20260906T100000Z` z logami i wynikami z przebiegu. 
W trakcie podaje też podstawowe informacje o postępie, żeby można było zobaczyć czy w ogóle działa jak powinien, czy warto już zatrzymać i poprawić ustawienia.


## Szczegóły funkcjonalności

### Jak to wszystko działa?

Po uruchomieniu następuje podstawowe sprawdzenie dopasowania serwera i modelu - tzw preflight. Wyniki z tego sprawdzenia pokazują czy w ogóle warto uruchamiać testy, bo niektóre pliki modeli mogą być uszkodzone, mieć uszkodzone template'y wywoływania narzędzi harnessu, serwer może nie wspierać niektórych kombinacji opcji z OpenAI API - i wtedy warto to wiedzieć żeby nie oceniać modelu przez perspektywę ograniczeń samego serwera.

Jest to sprawdzenie, czy serwer wstawia `tools` do promptu (`prompt_tokens` z narzędziami i bez). Brak różnicy - te przypadki są pomijane (wada szablonu/serwera). Wynik trafia do `preflight.json`.

Pojedynczy test to jedno zadanie z katalogu zadań. Uruchomione raz w środowisku: jeden model, jeden profil samplera (temperatura, seed...), jeden wariant promptu (neutral, helpful...) i jeden numer powtórzenia. Bench składa rozmowę — opcjonalną wiadomość `system` z wybranego wariantu, potem treść zadania jako `user` — i dokłada katalog narzędzi, które model może wywołać. Albo czasem pustą listę. Całość idzie jednym żądaniem `POST /v1/chat/completions` na serwer hostujący modele, z powyższymi parametrami.

Jeśli model w odpowiedzi zwraca `tool_calls`, harness nie woła prawdziwej pogody ani kalendarza. Odpalany jest lokalny mock i wynik wraca do modelu jako wiadomość `role: tool`. W suicie `tools` większość przypadków kończy się na tej jednej turze: model albo woła narzędzie, albo odmawia, albo pisze zwykły tekst. Część testów narzędziowych i cała suita `agent` powtarza pętlę — model, mock, znowu model — aż asystent przestanie wołać narzędzia albo skończy się limit kroków. Suita `coding` nie używa narzędzi: jest jedna długa wiadomość użytkownika i jedna (również długa) odpowiedź asystenta.

Gdy rozmowa się kończy, bench wystawia twarde 0/1 (no i ewentualne błędy jakie napotka). Sprawdza mechanicznie to, co da się sprawdzić: czy argumenty to poprawny JSON, czy nazwa narzędzia jest z katalogu, czy typy i wartości `enum` się zgadzają, czy w tekście dla użytkownika nie wyciekł szablon wywołania, czy w finale jest unikalny token z mocka, gdy test tego wymaga. Transkrypt i wynik lądują w plikach `trial_NNN.json` / `trial_NNN.txt`, a na konsoli widać ok albo fail.

Potem bench przechodzi do kolejnego zadania.

Po zakończeniu przebiegu bench zapisuje wyniki.

Po skończeniu działania benchu i zapisaniu wyników następuje etap ich sprawdzenia - rola sędziego - ewaluacji efektów pracy modelu i na jej podstawie korekcji liczb wynikających z twardych asercji testów. Ma na celu logiczno, językową weryfikację zachowania i efektów zwróconych przez testowany model w danym zadaniu przy zadanych parametrach pracy. Kilka powtórzeń pozwala również na ewaluację powtarzalności zachownia modelu.


### Domyślne parametry uruchamiania

Domyślne parametry są dobre do przetestowania czy w ogóle setup działa. Komenda na start znajduje się poniżej i używa następujących parametrów domyślnych:

```text
python run.py run
```

config: ``--config config.yaml`
suita: `--suites tools`
profil: `--profiles greedy`

Pełne testy warto uruchomić dopiero jak już każdy z wymienionych w pliku konfiguracyjnym model został sprawdzony na tych ustawieniach. Opcja `--verbose` wypisuje każde oceniane wywołanie HTTP do serwera modeli jako `\n\nRequest:\n` plus surowe ciało POST po wysłaniu, potem `\n\nResponse:\n` plus pełne ciało odpowiedzi po odbiorze. Nie dotyczy preflight ani `summarize`.

Komenda do skopiowania i usunięcia wartości z list, czy parametrów których nie potrzebujesz, jak już podstawowa działa. Zapoznaj się z detalami opcji w poniższych sekcjach.

```text
python run.py run --verbose --config my.config.yaml --profiles greedy,real --suites tools,agent,coding --prompt-variants neutral,helpful,instructed,harness --out results
```

- `harness` — użyj tylko wtedy, gdy w configu yaml jest uzupełniony system prompt, wysyłany przez używanego agenta (np Hermes).


### Profile samplera

| Profil   | Sens                                                                            |
| -------- | ------------------------------------------------------------------------------- |
| `greedy` | `temperature: 0`, stały seed. Sanity check i detektor niedeterminizmu backendu. |
| `real`   | Temperatura brana jest z `recommended_temperature`, per model.                  |

Dla każdego profilu w konfigu definiuje się ilość powtórzeń danego testu. Powtórzenia są istotne, ponieważ modele co do zasady są niederministyczne, nawet mimo ustawienia `temperature: 0`.

Wszystkie pola samplera (`top_p`, `top_k`, `min_p`, `repeat_penalty`, thinking / `chat_template_kwargs`) są obowiązkowe i są kopiowane do manifestu w wynikach. Na początek nie zmieniaj wartości domyślnych.


### Warianty promptu

Nie mieszaj ich w jednej liczbie nagłówkowej. Ścieżka: `<model>/<profile>/<prompt_variant>/`.


| Wariant      | System prompt                         | Co mierzy                                                                      |
| ------------ | ------------------------------------- | ------------------------------------------------------------------------------ |
| `neutral`    | *(brak — bez wiadomości* `system`*)*  | Naturalne zachowanie modelu.                                                   |
| `helpful`    | `SYS_NEUTRAL`                         | Najprostszy system prompt - `You are a helpful assistant` :)                   |
| `instructed` | `SYS_INSTRUCTED`                      | Na ile zachowanie modelu może być wysterowane starannie przygotowanym promptem |
| `harness`    | `harness_system` w configu            | Tu możesz wpisać własny prompt swojego prawdziwego agenta.                     |

W przypadku wariantu `instructed` niezmiernie istotna jest treść system promptu. Niewielkie zmiany, niejednoznaczności bardzo mocno wpływają na zachowanie modelu. Najczęściej tutaj właśnie tkwią przyczyny błędów i niedopasowania zachowania modelu do oczekiwań użytkowników. Niestety system prompt nie rozwiązuje wszystkich błędów.

`delta_instructed_minus_neutral` (per wymiar i per case): dodatnia = rubryka naprawiła awarię. Wyciek wywołań narzędzi do tekstowej odpowiedzi dla użytkownika na profilu `neutral`, czysty na `instructed` oznacza model używalny przy dobrym system prompcie. Wyciek na obu oznacza że akurat tego zachowania raczej będzie się ciężko pozbyć z modelu.

Kolejne testy i tury w ramach testów wysyłają do modelu **zadanie i dane**, nigdy mechaniki oceny.

Domyślny config: `neutral` `helpful` i `instructed`. `harness` można użyć dopiero po wypełnieniu `harness_system` w konfigu.

```text
python run.py run --prompt-variants neutral
python run.py run --prompt-variants helpful
python run.py run --prompt-variants neutral,helpful,instructed
```


### Suity

Parametr `--suites` wybiera jakie typy testów model dostanie. Każdy typ ma zdefiniowaną ilość powtórzeń per suita oraz profil w configu.

| Suita    | Zawartość                                                               |
| -------- | ----------------------------------------------------------------------  |
| `tools`  | Pojedyncze testy, pytanie, sprawdzenie zachowania: T01, T02...          |
| `agent`  | Zadania, sprawdzenie zachowania, kilka tur: A01, A02...                 |
| `coding` | Zadanie programistyczne, efekt końcowy raczej do oceny _manualnej_: C01 |


#### Tools (T01–T18)

To najwygodniejsza suita na pierwszy przebieg: krótkie testy, zwykle jedna wymiana z modelem. Dostaje on pytanie użytkownika i listę narzędzi, z których może skorzystać. Narzędzia są fikcyjne — bench sam odpowiada na wywołania — więc nie ma wywołań sprawdzania prawdziwej pogody ani na przykład poczty. W odpowiedzi widać wtedy, czy model wybrał właściwą funkcję i czy w finale powtórzył wynik, czy zmyślił.

Szukamy tu czytelnych zachowań. Model ma wołać narzędzie tylko gdy jest potrzebne, brać argumenty z tego, co użytkownik naprawdę powiedział, nie zgadywać brakujących faktów i nie wklejać do czatu surowego wywołania narzędzia. Część zadań jest po polsku (gdy pada miasto, to Wrocław), część po angielsku (London). Testy bez polskiego odpowiednika nie mają sufiksu `_pl` / `_en`.

Nie musisz znać numerów na starcie — pojawią się w wynikach. Krótko, o co w nich chodzi:

| ID  | Co się dzieje |
| --- | ------------- |
| T01 | Użytkownik podaje miasto i prosi o pogodę. Model powinien wywołać narzędzie z tą samą nazwą. |
| T02 | Na stole jest osiem narzędzi. Trzeba wybrać jedno, pasujące do pytania. |
| T03 | Pytanie z wiedzy ogólnej. Narzędzi wołać nie wolno. |
| T04 | Prośba o pogodę, ale miasto nie pada w treści. Wpisanie jakiegokolwiek miasta to zgadywanie. |
| T05 | Argumenty muszą mieć właściwe typy: liczba jako liczba, tak/nie jako bool, wartość z listy. |
| T06 | Bardziej złożone argumenty (obiekt, lista, godzina). Dziewiąta rano to godzina 9, nie 21. |
| T07 | Dwa niezależne polecenia w jednej wypowiedzi — dwa wywołania, kolejność nieważna. |
| T08 | Tym razem narzędzi używać nie wolno. Oczekiwana jest zwykła odpowiedź tekstowa. |
| T09 | Model nie może wymyślić nazwy narzędzia, którego nie dostał. |
| T10 | Trzeba wiernie skopiować identyfikator z polskimi znakami, bez „poprawiania”. |
| T11 | Narzędzie bez argumentów — puste wywołanie, nic od siebie nie dopisywać. |
| T12 | Użytkownik podaje wartość spoza dozwolonej listy. Model nie powinien jej przepchnąć jako poprawnej. |
| T13 | Dzielnica jest opcjonalna i użytkownik jej nie podał. Nie dodawać jej samemu. |
| T14 | Dwa podobne narzędzia: pogoda teraz kontra prognoza. Trzeba wybrać to, o które pytano. |
| T15 | Tego samego wywołania nie robić dwa razy pod rząd bez potrzeby. |
| T16 | To samo co T01, tylko rozmowa jest już długa. Model nadal ma trafić w miasto. |
| T17 | Najpierw utwórz wydarzenie, potem wyślij mail. Mail na początku to błąd. |
| T18 | Dwie tury: najpierw wyszukaj miejsce, potem dopytaj o szczegóły, używając identyfikatora z wyniku. |


#### Agent (A01–A07)

Tu zadanie zwykle nie kończy się na jednym wywołaniu. Model może zawołać narzędzie, dostać wynik i kontynuować, aż odpowie użytkownikowi albo skończy się limit kroków. To jest bliższe temu, jak naprawdę wygląda pętla agenta.

Sprawdzamy, czy po odczytaniu wyniku potrafi się zatrzymać, czy wplata ten wynik w odpowiedź, czy po błędzie narzędzia nie udaje że „się udało”, i czy nie wykonuje nieodwracalnych akcji (kasowanie pliku, wipe dysku) bez wyraźnego potwierdzenia. Są wersje polskie i angielskie; A05 i A07 są tylko po angielsku.

| ID  | Co się dzieje |
| --- | ------------- |
| A01 | Odpowiedź końcowa ma zawierać konkretną wartość, którą narzędzie właśnie zwróciło. |
| A02 | Po odczytaniu temperatury koniec. Nieużywane narzędzie z katalogu nie może kręcić pętli w kółko. |
| A03 | Narzędzie zwraca błąd. Model ma poprawić wywołanie albo wyjaśnić problem, nie udawać sukcesu. |
| A04 | Prośba o skasowanie pliku bez potwierdzenia — nie wołać narzędzia. |
| A05 | Duży katalog, kilkanaście narzędzi. Trzeba nie zgubić się w liście. |
| A06 | Druga wypowiedź użytkownika korzysta z tego, co model już wcześniej odczytał. |
| A07 | Sprzeczne instrukcje (kopia zapasowa kontra skasowanie dysku). Model ma się zatrzymać i dopytać, nie rozstrzygać konfliktu sam. |


#### Kodowanie

Zadanie programistyczne. Model nie dostaje narzędzi: w jednej odpowiedzi ma napisać mały program — klient rozmowy z narzędziami, zgodnie ze specyfikacją w [`bench/coding/`](bench/coding/) — i w tej samej odpowiedzi sam ocenić, czy kod spełnia wymagania. Na końcu powinien zostawić wyraźny werdykt: udało się albo nie.

Bench tego programu nie uruchamia. Zapisuje całą rozmowę i wycięte fragmenty kodu oraz sprawdza tylko, czy to w ogóle wygląda na składny Python. Ocenę „czy to naprawdę działa i czy recenzja nie była samohymnem” zostawiasz sobie albo większemu modelowi. Ten test trwa długo; odpalaj go, gdy krótsze suity już chodzą.


### Opcjonalny sweep temperatury

W configu jest jeszcze opcja `optional_temp_sweep.enabled: true`. Jest ona przeznaczona tylko dla suity wymienionej w `optional_temp_sweep.suite`, domyślnie `tools` i `agents`. dodatkowych kilka temperatur. Wyniki umieszczane są w `temp_sweep.json`. Wyniki uzyskane z tej opcji nie mieszają się z wynikami opcji głównych.


### Wyniki

Po przebiegu bench tworzy katalog z datą, na przykład `bench/results/20260906T100000Z`. Tam ląduje wszystko, co jest potrzebne do oceny: transkrypty rozmów, surowy HTTP w `*.raw.txt`, warstwowe `summary.json`, i kopia konfiguracji. Wymagane limity ściany czasu na jedno wywołanie HTTP: `tools: 60s`, `agent: 90s`, `coding: 2400s`. Jeśli ten limit minie, a strumień jeszcze produkował treść w oknie `min(request_timeout_s / 2, 5s)` przed cutoffem, trial dostaje `CASE_GENERATION_TIMEOUT` (liczony jak max_tokens). Cichy strumień, request bez streamu albo nieudany TCP w `connect_timeout_s` (2s) to `INFRA_ERROR`. `max_tokens` tools/agent = 1024; C01 bierze `ctx_size` z launchera (domyślnie 16384).

Żeby przeliczyć podsumowania dla zestawu drzew modeli (bez ponownego odpalania modeli):

```text
mkdir results\zestaw-do-sedziego
xcopy /E /I results\20260908T045257Z\gemma-4-12b-it-Q4_K_M results\zestaw-do-sedziego\gemma-4-12b-it-Q4_K_M
xcopy /E /I results\20260909T065153Z\mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M results\zestaw-do-sedziego\mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M
python run.py summarize results\zestaw-do-sedziego
```

Jako punkt startowy są dwa miejsca: `summary.txt` mówi, co przeszło a co nie. W katalogach `cases/` leżą konkretne rozmowy — plik `.txt` czyta się jak czat. Gdy coś padnie, transkrypt pokazuje, czy model wołał narzędzie, zgadywał miasto, czy wkleił do odpowiedzi śmieci ze szablonu.

Reszta plików przydaje się później, przy porównywaniu modeli albo sprawdzaniu, czy zawinił serwer, a nie sam model. Suita `coding` zapisuje całą odpowiedź osobno, razem z wyciętymi kawałkami kodu.

Pełna struktura wyników:

```text
results/<timestamp>/
  MANIFEST.json
  config.snapshot.yaml
  summary.json
  summary.txt                # skrót: co przeszło, co nie
  trial_id_map.json
  INTERRUPTED.txt            # tylko po przerwaniu przebiegu
  <model>/<profile>/
    preflight.json
    ABORTED.txt              # gdy preflight nie przeszedł
    coding/trial_001/        # tylko gdy leciało kodowanie
      conversation.json
      conversation.txt
      conversation.raw.txt
      attempts/
      tool_client.py
      python_checks.json
      meta.json
      trial.json
    <prompt_variant>/
      SYSTEM.txt
      temp_sweep.json        # tylko przy włączonym sweepie
      cases/<id>/CASE.md
      cases/<id>/trial_001.{txt,json,raw.txt}
```


## Sędzia

Automatyczne 0/1 nie kończy oceny. Sędziemu dajesz katalog wyników (ma własne `README.md`, `judge/`, `CASE.md`, transkrypty). Nie oczekuj, że otworzy to repo. Mechaniczny ground truth to pliki trial plus `summary.json` wariantu promptu; podsumowania greedy/real, modelu i korzenia to średnie z nagłówków dzieci.


### Interpretacja

Łatwo pomylić wadę serwera z wadą modelu. Jeśli model jednocześnie nie woła narzędzia gdy trzeba, i woła je gdy nie wolno, najpierw sprawdź szablon czatu i flagę `--jinja`. Zły szablon psuje cały przebieg — wtedy nie ma sensu odrzucać modelu.

Dwa GGUF porównuj jako dwa osobne wpisy w `models` w configu, na tym samym serwerze i tych samych suitach.

Profil `greedy` (temperatura 0) jest po to, żeby zobaczyć czy w ogóle działa i czy backend nie losuje przy „wyłączonej” losowości. Do decyzji, czy model nadaje się na agenta, patrz na `real` i na to *jakich* błędów jest dużo, nie na jedną średnią.


## Zawartość projektu

| Ścieżka                                                              | Rola                                                                                          |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `[llama.bat.example](llama.bat.example)` / `[llama.sh.example](llama.sh.example)` | Szablon lokalnego launchera llama-server (`--models-dir`, `--ctx-size`, `--jinja` — wymagane do `tools`). Skopiuj do `llama.bat` albo `llama.sh` i edytuj; te pliki są gitignored. |
| `[ollama.bat.example](ollama.bat.example)` / `[ollama.sh.example](ollama.sh.example)` | Szablon lokalnego launchera `ollama serve` (`OLLAMA_HOST`, `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_API_KEY`). Skopiuj do `ollama.bat` albo `ollama.sh` i edytuj; te pliki są gitignored. |
| `[bench/](bench/)`                                                   | Cały benchmark: CLI, przypadki, mocki, sędzia (pliki, bez wołania API).                       |
| `[bench/config.yaml.example](bench/config.yaml.example)`             | Endpoint, modele, profile samplera, suity i liczba powtórzeń. Do skopiowania do `config.yaml` |
| `[bench/src/bench/](bench/src/bench/)`                               | Kod: klient, preflight, runner, twarde 0/1, pętla kodowania.                                  |
| `[bench/src/bench/suites/cases.py](bench/src/bench/suites/cases.py)` | Definicje T01–T18, A01–A07.                                                                   |
| `[bench/coding/](bench/coding/)`                                     | Spec i kontrakt API C01.                                                                      |
| `[bench/judge/](bench/judge/)`                                       | Prompt sędziego, kryteria, kody naruszeń, `schema.json`.                                      |
| `[bench/results/](bench/results/)`                                   | Wyniki uruchomień (gitignored).                                                               |


### Poza projektem

RAG, GUI, voice, research w internecie, safety red-team.
