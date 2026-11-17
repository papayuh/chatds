/* SPDX-License-Identifier: MIT
 * Product condition 24. c14 primary ordering with round3 fallbacks only.
 * Rendering waits for decode completion, so wrappers/repeat never expose uncased raw tokens.
 */
#include <stdio.h>
#include <string.h>
#include "dispatch.h"
#include "chatds_storage.h"
#include "textops.h"
#include "learned.h"
#include "ownerfix.h"
static const char *const acts[] = {"", "hello.", "you're welcome.", "goodbye.", "i'm doing well, thanks.", "ok.",
    "i'm chatds, a small model on a nintendo ds lite.", "i can help with facts, math, python, shell and text."};
static const char noans[] = "i haven't said anything yet. what do you want to ask?";
static const char clarify[] = "do you mean something specific? can you be more specific?";
static void copy(char *d, size_t cap, const char *s) { size_t n = strlen(s); if (n >= cap) n = cap - 1; memmove(d, s, n); d[n] = 0; }
static void strip_title(const char *s, char *out, size_t cap) {
    char full[64]; copy(full, sizeof full, s); size_t n = strlen(full);
    if (n && full[n - 1] == ')') { char *last = NULL; for (char *p = full; (p = strstr(p, " (")); p++) last = p; if (last) *last = 0; }
    copy(out, cap, full);
}
static int direct(c14_runtime *r, const cds4_result *e, char *out, size_t cap) {
    char a[48], b[48], v[112]; const char *s = e->value; int f = e->field;
    if (f == C4R_DEFINITION) { copy(out, cap, s); return 0; }
    if (f == C4R_DATE_RANGE) {
        copy(v, sizeof v, s); char *sl = strchr(v, '/');
        if (sl) {
            *sl++ = 0; if (cds4_date_text(b, sizeof b, sl)) return -1;
            if (!v[0]) snprintf(out, cap, "ended %s.", b);
            else { if (cds4_date_text(a, sizeof a, v)) return -1; snprintf(out, cap, "from %s to %s.", a, b); }
        } else { if (cds4_date_text(a, sizeof a, v)) return -1; snprintf(out, cap, "%s.", a); }
    } else if (f == C4R_BIRTH_DATE || f == C4R_DEATH_DATE) {
        cds4_rec rec;
        if (cds4_date_text(a, sizeof a, s) || cds4_record(&r->kb4, e->recoff, f == C4R_BIRTH_DATE ? C4R_BIRTH_PLACE : C4R_DEATH_PLACE, &rec)) return -1;
        int place = rec.found && rec.f.value[0] && !(rec.f.flags & 1);
        snprintf(out, cap, "%s %s%s%s.", f == C4R_BIRTH_DATE ? "born" : "died", a, place ? " in " : "", place ? rec.f.value : "");
    } else if (f == C4R_BIRTH_PLACE || f == C4R_DEATH_PLACE || f == C4R_LOCATION || f == C4R_CREATOR) {
        snprintf(out, cap, "%s%s.", f == C4R_BIRTH_PLACE ? "born in " : f == C4R_DEATH_PLACE ? "died in " : f == C4R_LOCATION ? "in " : "created by ", s);
    } else if (f == C4R_HEIGHT || f == C4R_LENGTH) snprintf(out, cap, "%s %s.", s, f == C4R_HEIGHT ? "tall" : "long");
    else if (f == C4R_OCCUPATION) {
        int c = s[0]; if (c >= 'A' && c <= 'Z') c += 32;
        snprintf(out, cap, "%s %s.", c && strchr("aeiou", c) ? "an" : "a", s);
    } else snprintf(out, cap, "%s.", s);
    return 0;
}
static int retrieve(c14_runtime *r, const char *q) {
    return cds4_retrieve((cds4_ctx *)r->scratch.retrieval.b, &r->kb4, q, 3, 1, 1, &r->evidence);
}
static void fixed(c14_runtime *r, const char *path, const char *text, const char *kind) {
    copy(r->path, sizeof r->path, path); copy(r->g->opts.text, r->g->opts.text_cap, text);
    copy(r->template_name, sizeof r->template_name, kind ? kind : ""); r->deterministic = 1;
}
static void state_reply(c14_runtime *r, const char *kind, const char *text, int keep) {
    r->state_row = 1; r->keep = keep; fixed(r, "template", text, kind);
}
static int ordinal(c14_runtime *r, uint32_t off, const char *kind) {
    cds4_rec rec; if (!r->kb4.f || cds4_record(&r->kb4, off, C4R_DEFINITION, &rec)) return -1;
    strip_title(rec.title, r->dlg.entity, sizeof r->dlg.entity);
    snprintf(r->g->opts.text, r->g->opts.text_cap, "%s: %s", rec.title, rec.def);
    copy(r->g->answer.ctx, sizeof r->g->answer.ctx, rec.def);
    r->state_row = r->deterministic = 1; copy(r->path, sizeof r->path, "kb_direct");
    copy(r->template_name, sizeof r->template_name, kind);
    return 0;
}
static int resolve(c14_runtime *r, const char *term, const char *q) {
    char plain[128], desc[128] = "", keys[3][256]; c14_plain(q, plain);
    int count = 0; for (char *p = plain; *p;) {
        char *e = strchr(p, ' '); if (e) *e = 0; count++;
        if (!c14_has(p, "the a an one of i mean it is about") && strcmp(p, term)) {
            if (desc[0]) strcat(desc, " ");
            strcat(desc, p);
        }
        if (!e) break;
        p = e + 1;
    }
    c14_plain(q, plain);
    if (count > 5 || !desc[0] || !strcmp(term, "other") || c14_cue(q) ||
        !strncmp(plain, "what ", 5) || !strncmp(plain, "who ", 4) || !strncmp(plain, "when ", 5) || !strncmp(plain, "where ", 6) || !strncmp(plain, "how ", 4)) return 0;
    copy(keys[0], sizeof keys[0], term); snprintf(keys[1], sizeof keys[1], "%s %s", term, desc); snprintf(keys[2], sizeof keys[2], "%s %s", desc, term);
    /* Pending resolution and ranked retrieval never overlap. Reuse the
     * aligned 3 KiB arena rather than put twelve records on the DTCM stack. */
    uint32_t seen[12]; int ns = 0, nb = -1;
    c14_identity *recs = r->scratch.identities;
    cds4_rec record;
    for (int k = 0; k < 3; k++) {
        cds4_cand cs[4]; int nc = cds4_lookup(&r->kb4, keys[k], cs); if (nc < 0) return -1;
        for (int i = 0; i < nc; i++) {
            int duplicate = 0; for (int j = 0; j < ns; j++) duplicate |= seen[j] == cs[i].recoff;
            if (duplicate) continue;
            seen[ns] = cs[i].recoff;
            if (cds4_record(&r->kb4, seen[ns], C4R_DEFINITION, &record)) return -1;
            copy(recs[ns].title, sizeof recs[ns].title, record.title);
            copy(recs[ns].def, sizeof recs[ns].def, record.def);
            char blob[256]; snprintf(blob, sizeof blob, "%s %s", recs[ns].title, recs[ns].def); cds_normalize(blob, blob, sizeof blob);
            if (nb < 0 && c14_has(blob, desc)) nb = ns;
            ns++;
        }
    }
    if (nb < 0) return 0;
    char base[64], other[64]; strip_title(recs[nb].title, base, sizeof base); cds_style_apply(CDS_STYLE_LOWER, base, base);
    r->dlg.ncand = 0;
    for (int i = 0; i < ns; i++) if (i != nb) {
        strip_title(recs[i].title, other, sizeof other); cds_style_apply(CDS_STYLE_LOWER, other, other);
        if (!strcmp(base, other)) { r->dlg.cand[0] = seen[i]; r->dlg.ncand = 1; break; }
    }
    if (ordinal(r, seen[nb], "resolve")) return -1;
    return 1;
}
/* Keep recheck buffers off unrelated dispatch call chains on 5.5 KiB DTCM stack. */
static __attribute__((noinline)) void pushback(c14_runtime *r) {
    char text[512], raw[320]; int got = 0;
    if (r->dlg.task == CDS_TASK_FACTS && r->dlg.lastq[0] && r->kb4.f && retrieve(r, r->dlg.lastq) == CDS4_ANSWER_PRESENT) {
        cds4_result *e = &r->evidence;
        if (direct(r, e, raw, sizeof raw) == 0) {
            if (e->field == C4R_BIRTH_DATE || e->field == C4R_DEATH_DATE) {
                char year[16]; size_t n = strcspn(e->value + (e->value[0] == '-'), "-") + (e->value[0] == '-');
                if (n < sizeof year) { memcpy(year, e->value, n); year[n] = 0; snprintf(raw, sizeof raw, "%s %s.", e->field == C4R_BIRTH_DATE ? "born" : "died", year); }
            }
            snprintf(text, sizeof text, "i checked again. %s", raw); got = 1;
        }
    } else if (r->dlg.task == CDS_TASK_CALC && (cds_casual_answer(r->dlg.lastq, NULL, raw, sizeof raw) || owner_calc(r->dlg.lastq, raw, sizeof raw))) {
        char *end = strstr(raw, ") = "); if (end) { *end = 0; snprintf(text, sizeof text, "let me check again. %s = %s", raw + 5, end + 4); got = 1; }
    } else if (r->dlg.task == CDS_TASK_TEXT && (owner_text(r->dlg.lastq, raw, sizeof raw, NULL) || cds_textops(r->dlg.lastq, raw, sizeof raw, NULL))) {
        snprintf(text, sizeof text, "let me check again. %s", raw); got = 1;
    }
    if (!got) snprintf(text, sizeof text, "sorry, i can't recheck that. what i said was: %s", r->dlg.answer);
    state_reply(r, "pushback", text, 1);
}
static int dialogue(c14_runtime *r) {
    cds_dlg *d = &r->dlg; int dk = cds_dlg_kind(r->q);
    if (!dk && r->dec.act == A_REPEAT) dk = CDS_DK_REPEAT;
    if (!dk && r->dec.act == A_PUSHBACK) dk = CDS_DK_PUSHBACK;
    uint32_t cand[2] = {d->cand[0], d->cand[1]}; int nc = d->ncand, pend = d->pend; char term[24]; copy(term, sizeof term, d->term);
    d->pend = d->ncand = 0; d->term[0] = 0; d->cand[0] = d->cand[1] = 0;
    if (dk == CDS_DK_REPEAT || dk == CDS_DK_PUSHBACK) {
        if (!d->answer[0]) state_reply(r, "noans", noans, 0);
        else if (dk == CDS_DK_REPEAT) state_reply(r, "repeat", d->answer, 0);
        else pushback(r);
        return 1;
    }
    if (dk == CDS_DK_CANCEL || dk == CDS_DK_STYLE_OFF) {
        if (dk == CDS_DK_CANCEL) cds_dlg_reset(d); else cds_dlg_set_style(d, 0);
        r->no_style = 1; state_reply(r, dk == CDS_DK_CANCEL ? "cancel" : "style_off", "ok.", 0); return 1;
    }
    if (dk == CDS_DK_FIRST || dk == CDS_DK_OTHER) {
        if (nc) { if (ordinal(r, cand[dk == CDS_DK_FIRST || nc == 1 ? 0 : 1], "ordinal")) return -1; }
        else state_reply(r, "clarify", clarify, 1);
        return 1;
    }
    if (pend && term[0] && r->kb4.f) { int res = resolve(r, term, r->q); if (res) return res; }
    if (r->dec.handler == H_FACTS && d->entity[0]) cds_dlg_subst_pronoun(r->q, d->entity, r->effective, sizeof r->effective);
    if (!r->effective[0]) copy(r->effective, sizeof r->effective, r->q);
    if (!strcmp(r->q, r->effective) && r->dec.handler == H_FACTS && c14_pronoun(r->q) && !c14_entity_words(r->q) && !d->entity[0] && !d->answer[0]) {
        state_reply(r, "noans", noans, 0); return 1;
    }
    return 0;
}
static int template(c14_runtime *r, int h, int act) {
    const char *text = NULL, *kind = NULL, *term = NULL; char tmp[256], termcopy[24];
    if (h == H_CONVERSATION && act >= A_GREETING && act <= A_CAPABILITIES) { text = acts[act]; kind = "act"; }
    else if (h == H_CONVERSATION && (act == A_COMPLIMENT || act == A_INSULT)) {
        text = act == A_COMPLIMENT ? "thanks." : "sorry. how can i help?"; kind = "act";
    } else if (h == H_CONVERSATION) { text = clarify; kind = "clarify"; }
    else if (h == H_CLARIFY) {
        term = c14_ambig(r->q); char plain[128]; c14_plain(r->q, plain);
        if (c14_has(plain, "python")) { term = "python"; text = "do you mean the language or the snake?"; kind = "clarify_python"; }
        else if (r->evidence.status == CDS4_AMBIGUOUS && r->evidence.n_alt) {
            cds4_rec other; if (cds4_record(&r->kb4, r->evidence.alt_recoff[0], C4R_DEFINITION, &other)) return -1;
            snprintf(tmp, sizeof tmp, "do you mean %s or %s?", r->evidence.title, other.title); cds_style_apply(CDS_STYLE_LOWER, tmp, tmp);
            text = tmp; kind = "clarify_candidates";
            if (!term) { strip_title(r->evidence.title, termcopy, sizeof termcopy); cds_style_apply(CDS_STYLE_LOWER, termcopy, termcopy); term = termcopy; }
        } else { text = clarify; kind = "clarify"; }
    } else if (h == H_UNSUPPORTED) { text = "i can't do that. i only do facts, math, python, shell and text."; kind = "unsupported"; }
    else if (h == H_UNCLEAR) { text = "i didn't catch that. could you rephrase?"; kind = "unclear"; }
    else { text = "i don't have that detail in my offline facts."; kind = "missing"; }
    fixed(r, "template", text, kind);
    r->dlg.pend = term != NULL; copy(r->dlg.term, sizeof r->dlg.term, term ? term : ""); r->dlg.ncand = 0;
    if (!strcmp(kind, "clarify_candidates")) { r->dlg.cand[0] = r->evidence.recoff; r->dlg.cand[1] = r->evidence.alt_recoff[0]; r->dlg.ncand = 2; }
    return 0;
}
static __attribute__((noinline)) int act_allowed(c14_runtime *r) {
    char expr[CDS_PS_EXPR_CAP];
    int h = r->dec.handler;
    if (strcmp(r->effective, r->q) || cds_dlg_kind(r->q) || r->dec.act ||
        (h >= H_PYTHON && h <= H_CONVERSATION) ||
        (h == H_FACTS && c24_primary_fact(r->q)) || c14_ambig(r->q) || cds_textops_guard(r->q)) return 0;
    int text = cds_textops(r->q, r->g->opts.text, r->g->opts.text_cap, NULL);
    r->g->opts.text[0] = 0;
    return !text && !cds_calc_parse(r->q, expr, sizeof expr);
}
static const int learned_acts[] = {A_NONE, A_GREETING, A_THANKS, A_BYE, A_IDENTITY,
    A_CAPABILITIES, A_HOW_ARE_YOU, A_COMPLIMENT, A_INSULT, A_REPEAT, A_PUSHBACK};
static __attribute__((noinline)) int literal_length(c14_runtime *r) {
    char literal[128], key[128];
    int kind = cds_textops_length_literal(r->effective, literal, sizeof literal);
    if (!kind) return 0;
    if (kind == 2) {
        /* No KB available is not proof of a non-entity. Lookup errors fail
         * closed, and even ambiguous identities retain their factual route. */
        if (!r->kb4.f) return 0;
        cds_normalize(literal, key, sizeof key);
        cds4_cand candidates[CDS4_CAND_MAX];
        int n = cds4_lookup(&r->kb4, key, candidates);
        if (!n && !strncmp(key, "the ", 4)) n = cds4_lookup(&r->kb4, key + 4, candidates);
        if (n < 0) return -1;
        if (n) return 0;
    }
    /* Preserve existing deterministic counts and learned textops below. If
     * neither accepts, use the existing len/code backend rather than KB facts. */
    r->dec.handler = H_PYTHON; return 1;
}
/* Re-derive context from its source, not a styled/recheck/social reply. */
static __attribute__((noinline)) int casual_turn(c14_runtime *r) {
    char previous[256] = "";
    if (r->dlg.task == CDS_TASK_CALC &&
        !cds_casual_answer(r->dlg.lastq, NULL, previous, sizeof previous))
        owner_calc(r->dlg.lastq, previous, sizeof previous);
    return cds_casual_answer(r->effective, previous, r->g->opts.text, r->g->opts.text_cap);
}
static int core(c14_runtime *r) {
    char fact[128]; int has_fact = owner_fact(r->q, fact, sizeof fact);
    r->dec = c14_rules(has_fact ? fact : r->q);
    if (owner_identity(r->q)) r->dec = (c14_decision){H_CONVERSATION, A_IDENTITY};
    int h = r->dec.handler;
    int dr = dialogue(r); if (dr) return dr < 0 ? -1 : 0;
    if (has_fact && !strcmp(r->effective, r->q)) copy(r->effective, sizeof r->effective, fact);
    if (casual_turn(r)) {
        const char *end = strstr(r->g->opts.text, ") = ");
        size_t n = (size_t)(end - r->g->opts.text - 5);
        memcpy(r->effective, r->g->opts.text + 5, n); r->effective[n] = 0;
        r->deterministic = 1; r->dec.handler = H_CALC; copy(r->path, sizeof r->path, "calculator");
        copy(r->g->answer.calc, sizeof r->g->answer.calc, r->g->opts.text); return 0;
    }
    int length = literal_length(r); if (length < 0) return -1;
    if (length) h = r->dec.handler;
    if (act_allowed(r)) {
        int act = c24_act(r->g->session, r->q, &r->act_margin);
        if (act < 0) return -1;
        if (act) {
            r->learned_act = act; r->dec = (c14_decision){H_CONVERSATION, learned_acts[act]}; h = H_CONVERSATION;
            if (act == R3_REPEAT || act == R3_PUSHBACK) {
                dr = dialogue(r); if (dr) return dr < 0 ? -1 : 0;
            }
        }
    }
    if (owner_text(r->effective, r->g->opts.text, r->g->opts.text_cap, NULL) ||
        cds_textops_guarded(r->effective, r->g->opts.text, r->g->opts.text_cap, NULL)) {
        r->deterministic = 1; copy(r->path, sizeof r->path, "text_exact"); return 0;
    }
    if (!cds_casual_nonmath(r->effective) && owner_calc(r->effective, r->g->opts.text, r->g->opts.text_cap)) {
        r->deterministic = 1; r->dec.handler = H_CALC; copy(r->path, sizeof r->path, "calculator");
        copy(r->g->answer.calc, sizeof r->g->answer.calc, r->g->opts.text); return 0;
    }
    int notebook = mnb_parse(r->effective, &r->mathnb);
    if (notebook) {
        r->mathnb_active = notebook > 0 && mnb_evaluate(&r->mathnb, &r->mathnb_values);
        if (r->mathnb_active)
            snprintf(r->g->opts.text, r->g->opts.text_cap, "Pocket math: %s (%u points).",
                     r->mathnb.kind == MNB_PLOT ? "plot" : r->mathnb.kind == MNB_BAR ? "bars" : "table",
                     r->mathnb.points);
        else copy(r->g->opts.text, r->g->opts.text_cap, MNB_REFUSAL);
        r->deterministic = 1; r->dec.handler = H_MATHNB;
        copy(r->path, sizeof r->path, "mathnb"); return 0;
    }
    if (!strcmp(r->effective, r->q)) {
        cds_round3_text result;
        int n = c24_text(r->g->session, r->q, &result, r->g->opts.text, r->g->opts.text_cap);
        if (n < 0) return -1;
        if (n) {
            r->learned_text = result.op + 1; r->op_margin = result.margin; r->tag_margin = result.tag_margin;
            r->deterministic = 1; r->dec.handler = H_TEXT; copy(r->path, sizeof r->path, "text_exact"); return 0;
        }
    }
    if (h >= H_CONVERSATION) return template(r, h, r->dec.act);
    if (h == H_FACTS && r->kb4.f) {
        if (c14_pronoun(r->effective) && !c14_entity_words(r->effective)) return template(r, H_CLARIFY, 0);
        int st = retrieve(r, r->effective);
        if (st == CDS4_IO_ERROR || st == CDS4_CORRUPT_KB) return -1;
        if (st == CDS4_ANSWER_PRESENT || st == CDS4_FIELD_MISSING) strip_title(r->evidence.title, r->dlg.entity, sizeof r->dlg.entity);
        if (st == CDS4_ANSWER_PRESENT) {
            if (direct(r, &r->evidence, r->g->opts.text, r->g->opts.text_cap)) return -1;
            copy(r->g->answer.ctx, sizeof r->g->answer.ctx, r->evidence.context);
            r->deterministic = 1; copy(r->path, sizeof r->path, "kb_direct"); return 0;
        }
        if (st == CDS4_AMBIGUOUS) return template(r, H_CLARIFY, 0);
        int digit = strpbrk(r->effective, "0123456789") != NULL;
        if (st == CDS4_FIELD_MISSING || (st == CDS4_NO_MATCH && c14_cue(r->effective) && !digit)) {
            char ws[128]; cds_normalize(r->effective, ws, sizeof ws);
            if (st == CDS4_NO_MATCH && (c14_has(ws, "you your yourself youre chatds") || (c14_has(ws, "talking") && c14_has(ws, "to")))) return template(r, H_CONVERSATION, A_IDENTITY);
            if (st == CDS4_FIELD_MISSING || (r->evidence.expansion && r->evidence.reason && !strcmp(r->evidence.reason, "no_candidate"))) return template(r, H_FACTS, 0);
        }
    } else if (h == H_FACTS && r->g->kb) {
        /* KB3 compatibility ONLY when kb4.bin absent. No fallback after corrupt KB4 or a KB4 miss. */
        cds_retrieve(r->g->kb, r->effective, r->g->answer.ctx);
    }
    copy(r->path, sizeof r->path, r->g->answer.ctx[0] ? "transformer_kb" : "transformer");
    if (cds_format_prompt(r->effective, r->g->answer.ctx, r->g->answer.prompt, sizeof r->g->answer.prompt) < 0) return -1;
    /* Same answer allowance as gen_host.c. Evidence trimming is bounded by context size. */
    int n;
    for (;;) {
        n = r->g->tok->encode(r->g->tok->ctx, r->g->answer.prompt, strlen(r->g->answer.prompt), 1, r->g->opts.ids, (int)r->g->opts.ids_cap);
        if (n <= (int)r->g->opts.ids_cap - 64 || !r->g->answer.ctx[0]) break;
        char *sp = strrchr(r->g->answer.ctx, ' '); if (sp) *sp = 0; else r->g->answer.ctx[0] = 0;
        cds_format_prompt(r->effective, r->g->answer.ctx, r->g->answer.prompt, sizeof r->g->answer.prompt);
    }
    /* Generator reset/forward owns the borrowed KV values from here on. */
    c24_models_invalidate(r->g->session);
    chatds_status st = cds_gen_begin(&r->g->run, r->g->session, r->g->tok, r->g->answer.prompt, &r->g->opts, &r->g->answer.gen);
    r->g->run.calc = r->g->answer.calc;
    return st == CHATDS_OK ? 0 : -1;
}
static void finish(c14_runtime *r) {
    char *text = r->g->opts.text;
    if (!r->deterministic) {
        if (r->g->answer.calc[0]) {
            if (!cds_casual_nonmath(r->effective) && owner_grounded(r->effective, r->g->answer.calc)) {
                copy(text, r->g->opts.text_cap, r->g->answer.calc); copy(r->path, sizeof r->path, "calculator");
            } else {
                r->g->answer.calc[0] = 0;
                copy(text, r->g->opts.text_cap, "which numbers and operation should i use?");
            }
        }
        size_t a = 0, n = strlen(text); for (size_t i = 0; i < n; i++) if (text[i] == '\n') text[i] = ' ';
        while (a < n && (unsigned char)text[a] <= 32) a++;
        while (n > a && (unsigned char)text[n - 1] <= 32) n--;
        memmove(text, text + a, n - a); text[n - a] = 0;
    }
    int task = r->dec.handler == H_FACTS ? CDS_TASK_FACTS : r->dec.handler == H_CALC ? CDS_TASK_CALC : r->dec.handler == H_TEXT ? CDS_TASK_TEXT : r->dec.handler == H_CLARIFY ? CDS_TASK_NONE : CDS_TASK_OTHER;
    if (!strcmp(r->path, "calculator")) task = CDS_TASK_CALC;
    if (!strcmp(r->path, "text_exact")) task = CDS_TASK_TEXT;
    if (!r->state_row && r->dec.handler != H_CONVERSATION) cds_dlg_note(&r->dlg, task, "", text, r->effective[0] ? r->effective : r->q, 1, r->keep);
    else if (r->keep) copy(r->dlg.answer, sizeof r->dlg.answer, text);
    int style = r->style;
    if (!style && !r->no_style && strcmp(r->template_name, "repeat")) style = cds_dlg_style_take(&r->dlg);
    if (style) {
        cds_style_apply(style, text, text);
        char path[40]; snprintf(path, sizeof path, "style+%.32s", r->path); copy(r->path, sizeof r->path, path);
        if (strcmp(r->template_name, "repeat") && strcmp(r->template_name, "style_ask")) copy(r->dlg.answer, sizeof r->dlg.answer, text);
    }
    r->finalized = 1;
}
int c14_open(c14_runtime *r, chatds_ui_engine_gen *g, const char *kb4_path) {
    memset(r, 0, sizeof *r); r->g = g;
    cds4_open_status st = cds4_open(&r->kb4, kb4_path);
    return st == CDS4_OPEN_OK || st == CDS4_OPEN_MISSING ? 0 : -1;
}
void c14_close(c14_runtime *r) { cds4_close(&r->kb4); }
int c14_begin(c14_runtime *r, const char *question) {
    if (strlen(question) > 127) { cds_dlg_reset(&r->dlg); return -1; }
    memset(&r->g->answer, 0, sizeof r->g->answer); memset(&r->g->run, 0, sizeof r->g->run); r->g->run.done = 1;
    memset(&r->evidence, 0, sizeof r->evidence); r->evidence.status = CDS4_GATED;
    r->g->opts.text[0] = 0; r->g->cancelled = 0;
    r->style = r->state_row = r->no_style = r->deterministic = r->finalized = 0; r->keep = 1; r->shown = 0;
    r->mathnb_active = 0;
    r->effective[0] = r->path[0] = r->template_name[0] = 0;
    r->learned_act = r->learned_text = 0; r->act_margin = r->op_margin = r->tag_margin = 0;
    r->dec = (c14_decision){H_CONVERSATION, A_NONE};
    copy(r->q, sizeof r->q, question);
    int kind = cds_style_detect(question, &r->style, r->inner, sizeof r->inner);
    if ((kind == CDS_ST_WRAP || kind == CDS_ST_PREFIX || kind == CDS_ST_SUFFIX || kind == CDS_ST_LITERAL) &&
        (cds_textops_guard(question) || (kind != CDS_ST_LITERAL && cds_textops(question, r->g->opts.text, r->g->opts.text_cap, NULL)))) kind = 0;
    r->g->opts.text[0] = 0;
    if (!kind) r->style = 0;
    if (kind == CDS_ST_FREE) {
        cds_dlg_set_style(&r->dlg, r->style); r->keep = 0; r->state_row = 1;
        fixed(r, "ask", "ok. what should i say?", "style_ask"); finish(r); return 0;
    }
    if (kind == CDS_ST_LITERAL) {
        r->state_row = 1; fixed(r, "literal", r->inner, NULL); finish(r); return 0;
    }
    if (kind) copy(r->q, sizeof r->q, r->inner);
    int bad = core(r);
    if (bad) { cds_dlg_reset(&r->dlg); r->g->run.st = CHATDS_BAD_FORMAT; return -1; }
    if (r->deterministic) finish(r);
    return 0;
}
chatds_gen_event c14_step(c14_runtime *r, char *out, size_t cap, size_t *len) {
    *len = 0;
    if (!r->finalized) {
        int more = cds_gen_step(&r->g->run);
        if (more) return CHATDS_GEN_PROGRESS;
        if (r->g->run.st != CHATDS_OK) { cds_dlg_reset(&r->dlg); return CHATDS_GEN_ERROR; }
        finish(r);
    }
    size_t n = strlen(r->g->opts.text) - r->shown;
    if (n) { if (n > cap) n = cap; memcpy(out, r->g->opts.text + r->shown, n); r->shown += n; *len = n; return CHATDS_GEN_TEXT; }
    return CHATDS_GEN_DONE;
}
void c14_cancel(c14_runtime *r) {
    r->g->cancelled = 1;
    if (!r->finalized) { cds_gen_step(&r->g->run); finish(r); }
    r->g->answer.gen.stop = CDS_STOP_CANCELLED;
    cds_dlg_reset(&r->dlg);
}
int c14_log(c14_runtime *r, const char *directory) {
    char path[256];
    if (cds_path_join(path, sizeof path, directory, "debug.log"))
        return cds_storage_error("log path failed in directory", directory);
    FILE *f = fopen(path, "a"); if (!f) return cds_storage_error("log open failed", path);
    fprintf(f, "path=%s handler=%d template=%s retrieval=%s field=%s lookups=%lu reads=%lu stop=%s learned_act=%d act_margin=%ld learned_text=%d op_margin=%ld tag_margin=%ld\n",
        r->path, r->dec.handler, r->template_name, cds4_status_name[r->evidence.status], cds4_rel_name[r->evidence.field],
        (unsigned long)r->evidence.lookups, (unsigned long)r->evidence.sector_reads, cds_stop_string(r->g->answer.gen.stop),
        r->learned_act, (long)r->act_margin, r->learned_text, (long)r->op_margin, (long)r->tag_margin);
    int error = ferror(f) ? (errno ? errno : EIO) : 0;
    if (fclose(f) && !error) error = errno ? errno : EIO;
    if (error) { errno = error; return cds_storage_error("log write failed", path); }
    return 0;
}
