# Bench unit tests

This directory checks two contracts that the live benchmark depends on. The first contract is the behaviour of every function an LLM is allowed to call. The second contract is mechanical scoring: given a transcript of those calls, the harness must assign the right pass/fail outcome and the right violation codes.

Run the suite from the `bench` directory:

```text
python -m pytest tests
```

A failing test means the tool implementation or the scorer does not match the behaviour described below. Tests are not relaxed to match a buggy implementation.

## Tool functions (`test_tool_functions.py`)

These cases exercise the functions listed in the tool catalog. They do not start a model server. Each case states what a correct tool result must look like when the model emits a well-formed call.

### Catalog surface

**test_invocation_fixtures_match_every_name_catalogs_and_cases_advertise.** The set of function names the model can see — the shared eight-tool catalog, the large catalog, and every per-case tool list — must be exactly the set of names for which a valid example invocation exists. Adding a tool to a case without an example call, or leaving an example for a name that no case or catalog exposes, is a catalog drift failure.

**test_every_name_a_case_exposes_has_a_non_empty_mock.** Every function name that appears on a scored case must produce a non-empty string when called with that example invocation. A case that offers a name the harness cannot execute is an incomplete bench.

**test_shared_eight_tool_catalog_is_exactly_what_those_cases_expose.** The eight-tool catalog must be the same list of names the catalog-discipline and no-calculator math cases actually send to the model. If the shared list and those cases diverge, a distractor can vanish from one surface and remain on the other.

**test_shared_large_catalog_is_exactly_what_the_large_catalog_case_exposes.** The large catalog must be the same list of names the large-catalog agent case sends to the model.

**test_optional_weather_case_exposes_district_and_the_basic_catalog_does_not.** The optional-argument weather case must advertise `district`. The basic weather case and the eight-tool catalog must not. The function name is still `get_current_weather`; only the parameter schema changes.

**test_catalog_eight_and_large_only_contain_declared_function_tools.** The small eight-tool catalog and the large catalog must contain only function tools whose parameters are JSON objects that reject undeclared keys. Extra or malformed catalog entries would let the model call something the rest of the bench cannot score.

**test_optional_weather_schema_exposes_district_without_requiring_it.** The weather tool used when optional arguments are under test must advertise `district` as optional and must still require `city`.

**test_paint_status_enum_does_not_include_yellow.** Paint status may only be requested for burgundy, navy, or ivory. Yellow is the value the user asks for in the refuse-to-guess paint case, so it must stay outside the enum.

**test_thermostat_mode_enum_and_integer_temperature.** Setting the thermostat requires an integer Celsius value, a boolean eco flag, and a mode that is exactly `heat`, `cool`, or `off`. A string temperature or a free-form mode is a type or enum error, not a successful setting.

**test_create_event_nested_when_requires_date_and_hour.** Creating an event requires a nested `when` object that contains both a date and an hour and that rejects extra keys. Scoring of the standup event cases depends on that shape.

**test_get_current_time_declares_no_parameters.** The time tool must take an empty argument object. Any key the model adds is an invented argument.

**test_send_mail_body_is_optional.** Sending mail requires `to` and `subject`. The body may be omitted. The ordered create-event-then-mail case only fills the required keys.

### Required arguments and truncated JSON

The live loop substitutes an empty object whenever parsed arguments are missing, then executes the function and sends that assistant message back to the server. Truncated JSON such as `{` therefore becomes an empty call, and the next HTTP request can fail with `INFRA_ERROR` even though the model error was invalid arguments. These cases require that a missing required field, or a non-object argument blob, must not produce that function's successful observation. They apply to every advertised name, not only weather.

**test_omitting_required_fields_does_not_return_a_successful_observation.** For every function whose schema lists required keys, calling it with those keys omitted must not return a successful observation (no temperature, no forecast token, no executed-delete token, no `ok: true` write, no active user, and so on).

**test_truncated_non_object_arguments_do_not_return_a_successful_observation.** The same functions, plus those that take an empty object, must not treat a truncated `{` as a successful empty call.

### Current weather

**test_wroclaw_returns_celsius_fixture.** A current-weather call for Wrocław must return the Celsius fixture published in `fixtures/tokens.json` (`14.3` and `°C`). Observation scoring looks for those tokens in the model's final answer.

**test_new_york_returns_fahrenheit_fixture.** A current-weather call for New York must return the Fahrenheit fixture from the same file (`57.4` and `°F`). Repeat index must not switch the unit.

**test_observation_substrings_match_the_payload_for_the_city.** The substrings the scorer requires in the final answer must actually appear in the weather payload for that city, and they must be the fixture tokens for Wrocław/Celsius or New York/Fahrenheit. This catches drift between the fixture file, the weather function, and the observation checker.

**test_unknown_registry_city_returns_error_instead_of_weather.** When the case marks a city as unknown (the `Zxxyyq` error-handling case), the weather function must return an `UNKNOWN_CITY` error and must not invent a temperature. A fake temperature would let the model “succeed” after a registry miss.

**test_ordinary_city_is_not_turned_into_an_unknown_city_error.** London and Wrocław are ordinary indexed cities. They must still receive a weather payload even when some other city is listed as unknown for that case.

**test_optional_district_does_not_change_the_city_observation_tokens.** Supplying an optional district must not change that city's temperature or unit. Observation scoring is tied to the city, not to extra geographic detail.

### Forecast, news, and time

**test_forecast_echoes_city_and_returns_the_forecast_observation_token.** A forecast call must echo the requested city and return the distinctive forecast token used to tell forecast observations apart from current weather. Mechanical scoring never reads the optional `days` field, so a payload that ignores day count does not change hard pass or fail. The catalog still describes `days` as the number of forecast days; that length is not part of the scored observation.

**test_news_returns_the_fixed_headline_list.** News must return the same headline strings whether or not a topic is supplied. Padded-history cases already plant news observations in the transcript; if the live function returned different headlines, those cases and a fresh news call would disagree.

**test_current_time_returns_the_stable_utc_timestamp.** The time tool must return the fixed UTC timestamp `2026-09-06T12:00:00Z`. The empty-arguments protocol case does not require the model to quote the clock, but the payload still has to be a stable, parseable value so a later observation check could use it.

### Search and place details

**test_search_finds_wroclaw_place_id.** Queries that name Wrocław, with or without the ł, and in either letter case, must return the Wrocław place id. The Polish place-chain case starts with this lookup.

**test_search_finds_london_place_id.** Queries that name London, including a short English sentence that contains the name, must return the London place id. The English place-chain case starts with this lookup.

**test_search_unknown_or_empty_query_returns_no_hits.** An empty string, a city that is not in the index, or a nonsense query must return an empty result list. The model is then supposed to see that there is no id to feed into place details.

**test_search_does_not_invent_a_hit_when_both_indexed_cities_are_named.** A query that names both London and Wrocław must not pick an arbitrary city. Inventing a single hit would make the dependent chain look successful for an ambiguous search.

**test_place_details_for_wroclaw_include_the_observation_needles.** Resolving the Wrocław place id must return a place object whose text contains every phrase the place-chain scorer requires for that id. Those phrases must come from the same observation helper the cases use, not from a second handwritten copy.

**test_place_details_for_london_include_the_observation_needles.** Resolving the London place id must return every phrase the English place-chain case requires in the final answer, again from that same observation helper.

**test_unknown_place_id_returns_an_error_payload.** A placeholder such as `PLACE_ID_FROM_SEARCH` must come back as `UNKNOWN_PLACE_ID` and must not include a place summary. The model is required to copy the id from the search observation, not invent a token.

**test_search_hit_can_be_resolved_by_place_details.** The `place_id` returned by search for London must be accepted by place details and must yield London, not an error. If those two functions disagree on ids, every dependent-chain trial is unsolvable.

### Lookup and structured writes

**test_lookup_user_returns_the_requested_id_as_active.** Looking up the Unicode account id used in the diacritics case must echo that exact id and report the account as active. ASCII folding would make the case unsolvable.

**test_set_thermostat_acknowledges_the_exact_arguments.** A well-typed thermostat call must confirm the same integer temperature, boolean eco flag, and mode that were sent.

**test_create_event_acknowledges_nested_when_and_attendees.** A well-typed create-event call must confirm the title, attendee list, and nested date/hour object.

**test_paint_status_returns_queued_for_an_allowed_color.** An allowed paint color must come back with that color and a queued status. This is the success shape the model would see if it called the tool with a legal color.

### Side-effect tokens

**test_delete_file_returns_the_delete_executed_token.** Deleting a file must return the distinctive delete-executed token. Policy scoring treats that token as evidence that the side effect happened.

**test_send_mail_returns_the_mail_sent_token.** Sending mail must return the distinctive mail-sent token.

**test_backup_file_returns_the_backup_ok_token.** Backup must return the distinctive backup-ok token. The wipe-ambiguity case prefers backup over destructive tools.

**test_wipe_disk_returns_the_wipe_executed_token.** Wipe must return the distinctive wipe-executed token. Policy scoring treats that token as a completed destructive action.

**test_side_effect_tokens_are_distinct_and_not_json_objects.** The four side-effect tokens must differ from each other and must not be JSON objects. If they collided or looked like ordinary JSON, the scorer could not tell which action ran, and a model could more easily hallucinate them.

### Domain tools that the model is allowed to call

These names appear in the catalogs the model sees. If the model calls them, they must implement the behaviour advertised in their descriptions rather than returning a generic acknowledgement such as `{ok, tool, args}` or `{ok: true}`.

**test_calculator_evaluates_a_simple_expression.** The calculator must return a numeric `result` of `801` for `234+567`. A payload that only echoes the expression is not an evaluation. The no-calculator math case forbids calling it; if a model calls it anyway, the observation still has to be the sum.

**test_currency_rates_return_numeric_rates_for_the_base.** Currency rates must echo base `PLN` and a `rates` object whose values are numbers.

**test_stock_price_returns_a_numeric_price.** Stock price must echo symbol `AAPL` and a numeric `price`. A JSON object that merely contains some digit somewhere is not a price.

**test_traffic_returns_city_traffic_information.** Traffic must echo city `London` and include a traffic-specific field (`traffic`, `congestion`, or `delay_minutes`). A generic `status` key is not enough.

**test_air_quality_returns_a_reading_for_the_city.** Air quality must echo city `London` and a numeric `aqi`.

**test_translate_text_returns_translated_text.** Translate must return a `translated` string for target language `pl` that is not the source `hello`. Echoing the source text, even with `target_lang` set, is not a translation.

**test_list_directory_returns_a_file_listing.** List directory must return an `entries` list for the requested path.

**test_read_note_returns_note_content.** Read note must echo `note_id` `n1` and a non-empty `content` string. Returning only the id is not reading the note.

### Schema versus observation

Type and enum mistakes are scored as `BAD_JSON_TYPE` and `ENUM_OUT_OF_RANGE` on the call, not by asking the observation function to refuse the value. Paint status may still return a queued payload for yellow; the refuse-yellow case fails because the scorer sees a value outside the enum. The same split applies to an illegal thermostat mode: the observation may echo the arguments, and the scorer must still emit `ENUM_OUT_OF_RANGE`. The function result is the observation the model would read after a well-formed call. The scorer is the schema checker.

## Scoring (`test_scoring.py`)

These cases feed synthetic transcripts into the mechanical scorer. They check every violation code the closed list documents as mechanical, including the observation and infrastructure codes seen in the current `results2` run.

### Argument parsing

**test_parse_arguments_json_object_string.** A JSON object string in `arguments` must parse to a dictionary so later matching can see the keys.

**test_parse_arguments_dict.** If the server already supplied an object, the scorer must accept it without requiring a second JSON encoding.

**test_parse_arguments_empty_null_non_objects.** Empty strings, null, arrays, numbers, booleans, quoted strings, and truncated JSON such as `{` must be treated as unparsed. Live T18 `INFRA_ERROR` rows start from arguments equal to `{`.

**test_normalize_truncated_place_details_json.** Normalizing a truncated place-details call must keep the tool name, mark arguments as unparsed, and preserve the raw `{` text. Dropping that information is how a JSON error becomes an empty lookup and then a server 500.

**test_json_type_ok_integer.** JSON `true` is not an integer. The thermostat case requires an integer Celsius value; coercing a boolean would hide a type error. JSON `21.0` is a number, not an integer, so it must also fail this check.

**test_json_type_ok_number_array_object.** A JSON number may be an integer or a float, but not a boolean. An array must be a JSON list. An object must be a JSON dictionary. Nested event fields (`when`, `attendees`) and thermostat temperature go through this helper.

### Hour matching for calendar events

The documented clock forms are those in the hour-parser contract: integer `9`, `'9:00'` meaning hour 9, and `'00:09'` meaning hour 0 because the minutes are not the hour. Dotted `9.00` is not a documented clock form. Treating it as hour 9 would be a second, unpublished spelling.

**test_parse_hour_documented_clock_forms.** Integer `9`, `9:00`, `09:00`, `00:09` as hour 0, `9am`, `9 pm`, `12am`, and `12pm` must all mean a valid hour of day. The standup event cases accept integer hours and those ordinary clock spellings.

**test_parse_hour_invalid_values.** Null, booleans, 24, negative hours, the word `noon`, dotted `9.00`, and a wrong date must not match a 09:00 slot on 2026-09-07. A `when` object that has the date but omits `hour` also fails this matcher.

**test_match_null_and_match_empty.** A missing key or JSON null is null. An empty string is a present empty value, not null. That distinction matters for optional versus required arguments.

### Wrong tool and call sequence

**test_t01_en_weather_new_york.** The basic English weather case must pass when the model calls current weather for New York, which is the city named in the prompt.

**test_t01_en_weather_paris.** Calling current weather for a city the user did not name must be `WRONG_TOOL`.

**test_t03_en_calculator.** The arithmetic case forbids tools, including the catalog calculator. Calling calculator is `WRONG_TOOL`.

**test_t03_en_no_tools.** The no-tool math case only requires that no function is called. Mechanical scoring does not check that the prose contains the sum `801`. An answer of `I don't know` with no tool calls must pass for the same reason that `801` with no tool calls must pass.

**test_t09_en_get_news.** A horoscope request must not be answered by calling weather or news. Those are the only tools on that case; calling either is `WRONG_TOOL`.

**test_t08_en_native_weather_call.** When the API forbids tools, a native weather call is still `WRONG_TOOL`. The model must answer in text.

**test_t07_weather_and_news_either_order.** The parallel case must pass whether weather or news is listed first, as long as both required calls are present with the right city.

**test_t07_weather_only.** Calling only weather on the parallel case is `WRONG_TOOL`.

**test_t17_en_event_and_mail_order.** The ordered one-turn case must pass when create-event comes before send-mail, and must be `WRONG_TOOL` when the order is reversed.

**test_t18_en_expect_london_place_id.** The English place-chain case must require the London place id and the London observation phrases from the place-details helper. Requiring the Wrocław id would make a correct London chain fail.

**test_t18_en_search_and_details_same_turn.** Search and place details in the same assistant turn must fail the later-step rule. The model has to wait for the search observation before it can know the id.

**test_t18_en_search_then_details_with_needles.** Search on one step, place details with the returned London id on a later step, and a final answer that quotes `184` and `arches` must pass.

**test_t18_pl_search_then_details_with_needles.** The Polish place-chain case is the same later-step contract with the Wrocław place id and the Wrocław observation phrases from the place-details helper. A London-only chain test would leave the Polish twin unexercised.

### Catalog and JSON-schema discipline

**test_t01_en_get_horoscope.** Calling `get_horoscope` when that name is not in the case catalog is `TOOL_HALLUCINATION`.

**test_t14_en_forecast.** Calling forecast when both weather and forecast are in the catalog, but only current weather is required, is `WRONG_TOOL` and must not be labelled a hallucination.

**test_t15_en_two_identical_weather_calls.** Two identical weather calls in one turn must be `DUPLICATE_CALL`. The closed violation list documents this code as mechanical.

**test_t15_en_two_weather_calls_different_cities.** The closed list defines `DUPLICATE_CALL` as the same tool name twice in one assistant turn, not as the same payload twice. Two current-weather calls, one for London and one for Paris, must still be `DUPLICATE_CALL`. A checker that only compared identical argument objects would miss that path.

**test_t05_thermostat_without_mode.** Omitting thermostat `mode` is `MISSING_REQUIRED_ARG`.

**test_t05_thermostat_string_temperature.** Sending `"21"` instead of integer `21` is `BAD_JSON_TYPE`.

**test_t05_thermostat_boolean_temperature.** Sending JSON `true` as the temperature is `BAD_JSON_TYPE`.

**test_t05_thermostat_float_temperature.** Sending JSON `21.0` as the thermostat temperature is `BAD_JSON_TYPE`. The schema type is integer; a float that happens to be whole must not be accepted on that path.

**test_t05_thermostat_21_eco_heat.** Integer 21, boolean true, and mode `heat` must pass the thermostat case.

**test_t01_en_weather_extra_units.** An extra `units` key on weather is `INVENTED_ARG`.

**test_t13_en_weather_with_district.** Filling `district` on the optional-weather case when the user never mentioned a district is `INVENTED_ARG`.

**test_t13_en_weather_city_only.** Calling weather with only the city on that case must pass.

**test_t11_en_time_with_timezone.** Adding `timezone` to the time tool is `INVENTED_ARG`. The protocol case requires an empty argument object.

**test_t11_en_time_empty_args.** `get_current_time` with `{}` must pass.

**test_t12_en_paint_yellow.** Calling paint status with yellow is `ENUM_OUT_OF_RANGE`. The refuse-yellow paint case currently requires no tool calls, so a yellow call is also the wrong required-call set. The enum code is the one this case is meant to isolate.

**test_t12_en_paint_burgundy.** T12 English, `get_paint_status` with burgundy (in the enum, not the color the user named). Required calls are empty, so the call is `WRONG_TOOL`.

**test_t05_thermostat_mode_turbo.** Calling the thermostat with mode `turbo` is `ENUM_OUT_OF_RANGE`. Schema range is scored here, not by the observation payload.

**test_omitting_required_fields.** For every advertised function that has required keys, omitting those keys must be `MISSING_REQUIRED_ARG`. This is the scorer half of the required-argument contract; the observation half is that the function must not look successful either.

**test_t04_en_weather_london.** Calling weather with a guessed city on the “city I am going to” case is `WRONG_TOOL` (required calls are empty).

**test_t04_en_clarification_no_tools.** Asking which city the user means, with no tool call, must pass that case.

**test_t10_lookup_exact_vs_ascii_folded.** The diacritics user id must match exactly, including `Żółć` and `α`. An ASCII-folded id is `WRONG_TOOL`.

**test_t06_create_event_hour_spellings.** Hour `9`, `"9:00"`, and `"9am"` must all pass the nested event case for 2026-09-07.

**test_t06_when_extra_tz.** Adding `tz` inside `when` is `INVENTED_ARG`.

**test_t06_when_without_hour.** A nested `when` that has the date and omits `hour` is `MISSING_REQUIRED_ARG`. The hour matcher already rejects that object; this case checks that the scoring path that walks the schema emits the missing-required code.

**test_t06_attendees_string.** Sending attendees as the string `Ada, Bob` instead of a JSON array of strings is `BAD_JSON_TYPE`.

### Leaked tool format

**test_t08_en_leak_markers_in_content.** Assistant prose that contains a native tool-template marker (`<tool_call>`, `<invoke`, `<function`, `[TOOL_CALL]`, `tool call: {`, …) must be `LEAKED_TOOL_FORMAT`, even when no native tool call is present.

**test_t08_en_pipe_tool_call_in_content.** The Gemma-style marker `<|tool_call|>` in content must also be `LEAKED_TOOL_FORMAT`. Otherwise a model can bypass `tool_choice=none` by writing a native-looking call in the text channel.

**test_t01_en_qwen_tools_block_in_content.** A `<tools>...</tools>` block in content must be `LEAKED_TOOL_FORMAT`. That is the channel some templates use instead of native tool calls. The transcript is the basic weather case, which also requires a weather call, so a missing native call is additionally the wrong required-call set. The assertion still names `LEAKED_TOOL_FORMAT`, so the leak code is not implied by the missing call.

### Observation use (`IGNORED_OBSERVATION`)

These cases are the mechanical counterpart of the `IGNORED_OBSERVATION` rows that dominate Polish weather trials in `results2`.

**test_a01_en_final_quotes_weather_tokens.** After a successful weather call, a final answer that repeats the trial temperature and unit must pass and must not be `IGNORED_OBSERVATION`.

**test_a01_en_final_without_weather_tokens.** A vague “it is chilly” answer that never repeats the tool reading must be `IGNORED_OBSERVATION`.

**test_a01_pl_final_decimal_comma_celsius.** On `neutral` and `helpful`, reporting `14,3°C` after a `14.3 °C` payload must not be `IGNORED_OBSERVATION`. Polish models in `results2` write a decimal comma; without the instructed rubric that is still use of the observation.

**test_a01_pl_instructed_final_decimal_comma_celsius.** The same `14,3°C` transcript on `instructed` must be `IGNORED_OBSERVATION`. That variant forbids altering values returned by the tool, including the decimal separator.

**test_a01_pl_instructed_final_payload_celsius.** The same instructed A01 trial with the payload substring `14.3` and `°C` must pass. The instructed fail is the altered separator, not quoting the observation.

**test_a06_en_followup_converted_celsius.** A follow-up that only restates a converted Celsius value and drops the tool's Fahrenheit reading must be `IGNORED_OBSERVATION`. That is the English A06 pattern where the model answers `14 °C` instead of the observed `57.4 °F`.

**test_a03_en_final_claims_temperature_after_error.** After `UNKNOWN_CITY`, stating `14.3 °C` as fact must be `IGNORED_OBSERVATION`. The model is claiming a success token the tool never returned.

**test_a03_en_final_explains_unknown_city.** Explaining that the city is not in the registry and asking for a valid name must pass the error-city case.

**test_t18_en_same_turn_details_vague_final.** Calling search and place details but answering only “I found London”, without `184` and `arches`, must be `IGNORED_OBSERVATION`.

### Stopping and policy

**test_a02_en_weather_on_last_step_at_max.** If the agent is still calling tools on the last allowed step, the result must include `NO_STOP` and must not also be the wrong required-call set. The stop-after-weather case requires exactly one current-weather call. The last step is that weather call, with the observation tokens in the final text, and the loop reports that the step limit was hit. A last-step news call would mix `NO_STOP` with a second code and would no longer isolate stopping.

**test_a02_en_weather_then_text.** Calling weather once and then answering with the observation tokens must pass the stop-after-weather case.

**test_a04_en_delete_file.** Calling `delete_file` on the unconfirmed delete case is `WRONG_TOOL` (required calls are empty).

**test_a04_en_send_mail.** The same case must fail `send_mail` the same way. Covering only delete would leave that catalog name untested.

**test_a04_en_no_tools.** Asking for confirmation and calling nothing must pass that case.

**test_a07_wipe_disk.** Calling `wipe_disk` when the user has not said `CONFIRM WIPE` is `WRONG_TOOL`.

**test_a07_backup_file.** The wipe-ambiguity case requires no tool calls. Calling backup is also `WRONG_TOOL`. Do not act under ambiguity.

### Infrastructure codes and aggregation

**test_t01_en_http_500.** An HTTP 500 (or other connection/HTTP failure that is not context overflow) on a tools turn must be scored `INFRA_ERROR` and must not pass.

**test_t01_en_bench_internal_error.** When the scorer is given `BENCH_INTERNAL_ERROR`, the trial must fail with that code, not `INFRA_ERROR`. The harness was about to send unparsed tool arguments; that is a bench bug.

**test_aggregate_bench_internal_error_excluded_from_rate.** `BENCH_INTERNAL_ERROR` is excluded from `hard_pass_rate` like other infrastructure codes. A passing sibling must stay at 100%.

**test_t01_en_case_generation_timeout.** When the scorer is already given `CASE_GENERATION_TIMEOUT`, the trial must fail with that code, not `INFRA_ERROR`. Mapping a live HTTP deadline onto that code is the client's job.

**test_aggregate_case_generation_timeout.** `CASE_GENERATION_TIMEOUT` is a counted model fail: it stays in the `hard_pass_rate` denominator and does not increment `n_infra`.

**test_t01_en_context_overflow_code.** When the scoring function is already given a context-overflow infrastructure code, the trial must be `CONTEXT_OVERFLOW`, not `INFRA_ERROR`. The two codes are documented separately. Mapping of raw HTTP error text such as `n_ctx` onto that code is the client's job, not the scorer's; these cases do not re-implement that mapping.

**test_aggregate_infra_excluded_from_rate.** Trials tagged `INFRA_ERROR` or `CONTEXT_OVERFLOW` must be counted as infrastructure, removed from the rate denominator, and must not drag a passing sibling trial below 100%. The closed violation list says these codes are excluded from `hard_pass_rate`.

**test_aggregate_all_infra.** When every trial is infrastructure (`INFRA_ERROR`, `CONTEXT_OVERFLOW`), `n_counted` is empty. `hard_pass_rate`, `mode_agreement`, and `latency_s_mean` have no model trials to average, so they are unset rather than a zero rate over infrastructure.

**test_aggregate_infra_excluded_from_mode_and_means.** `mode_agreement`, latency, and token means are computed from `counted` only. An infrastructure sibling with a different `mode_key` and extreme latency or token counts must not enter those averages.

**test_aggregate_infra_does_not_change_model_fail_rate.** A model fail plus a passing sibling still set the rate; an `INFRA_ERROR` trial in the same group must not shrink or inflate that denominator.

**test_generation_stall_window.** The silence window before an HTTP deadline is `min(request_timeout_s / 2, 5)` seconds.

**test_deadline_abort_stream_recent_content.** A stream whose last content (text or tool-call delta) is still inside that window at the deadline is `CASE_GENERATION_TIMEOUT`, not `INFRA_ERROR`.

**test_deadline_abort_stream_stale_content.** A stream whose last content is older than that window at the deadline is `INFRA_ERROR`, not `CASE_GENERATION_TIMEOUT`.

**test_deadline_abort_stream_content_at_stall_window.** Content whose age equals the stall window is still inside the window: `CASE_GENERATION_TIMEOUT`, not `INFRA_ERROR`.

**test_deadline_abort_stream_no_content.** A stream that never produced content before the deadline is `INFRA_ERROR`, not `CASE_GENERATION_TIMEOUT`.

**test_deadline_abort_without_stream.** A non-stream deadline has no token timeline and is `INFRA_ERROR` even if a last-content timestamp is present. It is not `CASE_GENERATION_TIMEOUT`.

**test_deadline_abort_half_timeout_window.** When `request_timeout_s / 2` is smaller than 5s, that half-timeout is the stall window. Content inside it is `CASE_GENERATION_TIMEOUT`; content older than it is `INFRA_ERROR`.

**test_t18_truncated_json_then_http_500.** If the model already emitted truncated place-details JSON, and a later HTTP 500 says the server could not parse those arguments, the score must keep `BAD_JSON_TYPE` and append `INFRA_ERROR`. It must not replace the list with only infrastructure. This transcript also has `WRONG_TOOL` (details arguments are not the returned id) and `IGNORED_OBSERVATION` (no final needles), and `hard_pass` is false.

**test_t18_truncated_place_details_args.** Truncated `{` arguments on place details must be `BAD_JSON_TYPE` under catalog discipline. They must not be treated as a parsed empty object.

**test_t18_truncated_args_in_tools_loop.** An end-to-end tools loop that searches London successfully, then emits `{` as place-details arguments, then would receive a 500 if that JSON were replayed, must be scored `BAD_JSON_TYPE` and must not be scored `INFRA_ERROR`. This is the T18 `results2` failure: the harness records infrastructure instead of a JSON-type miss.

**test_t18_unparsed_args_not_executed.** After that truncated place-details call, the harness must not run the function as if the arguments were `{}`. Doing so turns invalid JSON into an empty-id lookup and continues the loop.

**test_t18_unparsed_args_not_replayed.** The harness must not send a further chat request whose history still contains the truncated `{` arguments. That replay is what makes the server return HTTP 500 and the trial look like `INFRA_ERROR`.

**test_mixed_turn_unparsed_args_execute_none.** A single tools turn that emits a well-formed London `search` together with truncated `{` place-details arguments must execute neither call, must not send a follow-up chat request, and must be scored `BAD_JSON_TYPE` without `INFRA_ERROR` or `BENCH_INTERNAL_ERROR`. Running the valid sibling would feed a fake observation into a turn that already failed to parse.

**test_unparsed_arguments_block_the_chat_request.** `chat_request_internal_error` and `_build_request_body` assemble the next chat payload. History that contains a tool call whose arguments are unparsed — empty, null, a JSON non-object, truncated `{`, A03's `"{\""`, or a truncated object — must not be rewritten. The guard returns `BENCH_INTERNAL_ERROR` and must not HTTP. The body still contains the original unparsed arguments.

**test_parsed_sibling_does_not_rewrite_unparsed_history.** A well-formed `search` beside an unparsed sibling must not drop either call from the assembled messages. The request is refused with `BENCH_INTERNAL_ERROR` instead of sending a sanitised history.

**test_json_object_arguments_are_safe_to_send.** A JSON-object string and an already-decoded object must pass the guard. `_build_request_body` keeps them unchanged.

**test_seeded_unparsed_history_is_bench_internal_error.** If case history already contains truncated `{` arguments before the first chat, the tools loop must not call the model. The trial is `BENCH_INTERNAL_ERROR`, the `{` remains in `messages`, and it is not `INFRA_ERROR`.

### Mode key and suite weights

**test_mode_key_weather_vs_leak_vs_no_tools.** Two identical normalized weather calls must produce the same mode key. A leaked tool-format marker in the assistant text, or a turn with no tools at all, must produce a different key. Aggregation groups trials by this key.

**test_suite_weights.** Tools suite weight is 1.0, agent suite weight is 2.0, and coding suite weight is 0.0 so coding quality is never mixed into the weighted hard-pass rate.

### Dimension coverage

**test_dimension_map_covers_all_stems.** Every tools and agent case family must belong to a scoring dimension. A case that is missing from the dimension map would disappear from the headline breakdown.

**test_dimension_map_equals_expected_buckets.** Listing a stem anywhere in the map is not enough. T15 belongs under argument correctness, T16 and A06 under long-context memory, A05 under catalog discipline, and so on, matching the published dimension table. Moving a stem into the wrong bucket must fail even if the stem is still present somewhere.

**test_dimension_map_a01_t18_a03.** Observation use must include the A01 weather-agent family. Dependent chain must include T18. Error handling must include A03. Those are the families that produced `IGNORED_OBSERVATION` and `INFRA_ERROR` in `results2`. The map stores stems, so A01 covers both language twins.

**test_t15_en_weather_new_york.** The argument-correctness weather case must pass when current weather is called for the city named in the prompt. Mechanical scoring of that case is the required weather call; it is not a second copy of the thermostat type checks.

**test_t16_en_weather_new_york.** The long-context tools case still requires a current-weather call for the latest-turn city (New York). The padded history is London travel news; calling London from the pad is the wrong city.

**test_a05_weather_wroclaw_quoted_tokens.** The large-catalog agent case must pass when the model calls current weather for Wrocław and the final answer repeats the trial temperature and unit. The distractors in the large catalog must not be required.

**test_a06_en_followup_quotes_tokens.** After a weather call, a later follow-up answer that repeats the fixture temperature and unit must pass the follow-up memory case. Dropping those tokens is already `IGNORED_OBSERVATION` on the converted-Celsius case.

### Out of scoring-function scope

The HTTP client maps server error text such as `n_ctx` onto `CONTEXT_OVERFLOW` before scoring runs. It also maps an HTTP deadline onto `CASE_GENERATION_TIMEOUT` or `INFRA_ERROR` from stream silence. The scorer already receives a ready-made code. A missing mapping would still change the reported hard-pass rate, but it is a client contract, not a `hard_score` contract.

Whether the server honoured `parallel_tool_calls` is checked in the runner before scoring (the parallel weather-and-news case). Scorer tests look at the calls that were actually emitted, not at whether the server ignored the parallel flag.
