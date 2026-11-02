# CLAUDE.md

## Pull requests and the `/code-review` checkbox

Every PR must carry this line, ticked, in its description:

```
- [x] I ran `/code-review` in Claude Code on this PR, and addressed or
      consciously accepted every finding.
```

Three rules.

**Never tick the box unless it is already true.** Not "will be true shortly",
not "am about to run it". True at the moment the text is written. If the review
has not run, the box stays unticked.

**Do not open the PR until you can tick it.** Run `/code-review` against the
branch (diffed against `main`), deal with what it says, and open the PR with the
box already ticked. Review, then open -- not open, then review.

**Paste the box in yourself.** `.github/pull_request_template.md` only populates
the body when the PR is opened through the web UI. `gh pr create --body` and
`--body-file` bypass the template entirely, so the box is simply absent and
there is nothing unticked to notice. If you open PRs with `gh`, the line is your
responsibility. (PR #24 shipped without it for exactly this reason.)

If you push further commits after opening, the box no longer describes what is
on the branch. Re-run `/code-review` and say so in the PR, or untick it. Nothing
else will catch this -- the ruleset does not dismiss anything on push.

### Why this matters more than it looks

The ruleset on the default branch does not require an approving review. The box
replaced that requirement, so it is the only review gate left, and nothing
enforces it today. A ticked box that is not true does not route around the gate,
it removes it, and leaves a record claiming otherwise.

To be accurate about the "nothing enforces it" part: that is a choice, not a
limitation. A `pull_request`-triggered workflow that greps the body for `- [x]`
can be marked a required status check, which would make the box a real gate. It
was considered and deliberately not added -- the repository has no CI at all.
If the honour system stops being enough, that is the thing to build.

It is also an easy rule to break by accident. Writing a PR body is the natural
moment to fill in a box describing something you intend to do next. That
happened here within eight minutes of the policy being adopted, which is what
prompted this file.

### What the box claims

"Addressed or consciously accepted", not "zero findings" -- a finding you
considered and disagreed with satisfies it, as long as the PR says so. What does
not satisfy it is not having looked. The template's own comment says the same
thing; if the two ever disagree, the template is the one people actually read
while filling in a PR, so fix this file to match it.
