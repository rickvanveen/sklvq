# Rules for test-suite PRs

Apply these rules when developing or reviewing this repository's test suite.

## Scope and assessment

- Read the implementation, its callers and existing tests before changing the suite.
  Identify the behavior that training, prediction or other real callers rely on.
- Test internal functions as internal functions. Do not build a general-purpose
  public API validation suite around inputs their callers do not supply.
- Cover each meaningful behavior once. Keep cases that exercise distinct
  mathematics, branches, parameter selection or confirmed defects. Avoid exhaustive
  combinations and repeated cases that establish the same behavior.
- Use coverage to find omissions, not as a target. Neither 100% coverage nor a
  large test count proves that the suite is useful.
- Keep test-suite PRs focused on testing. Do not change production behavior, add
  benchmarks, or repair unrelated configuration unless explicitly requested.
- Work on a dedicated branch. Inspect and preserve existing work, including
  changes from other sessions.

## What to test

- Anchor mathematical correctness with small, explicit inputs and hand-calculated
  expected answers. Explain the origin of non-obvious values.
- Avoid reimplementing the production formula as the default test oracle.
  Comparing a wrapper with the same dependency call it wraps adds little assurance.
- Check analytical gradients against finite differences of the relevant raw
  parameters, in the order callers consume them. Avoid accidental normalization
  during perturbation. Test nondifferentiable conventions separately.
- Select edge cases because the implementation or callers give them meaning:
  for example, local matrix selection, rectangular relevance, missing observations
  and one-sample training batches. A tiny synthetic shape alone is not a reason
  to add another test.
- Keep input-preservation checks where callers reuse samples or model parameters.
  Snapshot caller-owned arrays and compare them after each relevant call,
  including missing-value masking. Keep these checks independent of expected
  numerical failures.
- Add invariants only when they provide distinct protection beyond the chosen
  examples. Do not repeat a generic geometry checklist for every implementation.
- Test this library's behavior, not NumPy, SciPy or pytest internals. Leave
  user-facing validation and real model initialization/integration to the tests
  responsible for those boundaries.

## Structure and readability

- Prefer implementation-focused tests that make the function being tested clear.
  Parameterize genuinely shared behavior only when this preserves readability;
  do not hide every implementation behind one dynamic test framework.
- Use the smallest fixture that supplies the state a unit consumes.
  `SimpleNamespace` is sufficient when only a few attributes and methods are
  needed. Do not fit full models just to supply unit-test state, introduce a
  mocking dependency, or build a fake-model framework.
- Preserve real-model integration tests. Lightweight fixtures do not validate
  model initialization or the complete training/prediction path.
- Share only useful setup or numerical mechanics, such as finite differences.
  Keep the example, expected answer and assertion easy to find in the test.
- Name results clearly: `expected` and `actual`, or more specific names such as
  `expected_gradient` and `actual_gradient`. Where broadcasting could hide a wrong
  result shape, check the shape alongside the numerical comparison.
- Organize code into logical paragraphs, separated by one blank line.
  Put short comments immediately above the block they explain. Explain intent,
  mathematics or a pitfall; do not narrate obvious assignments or assertions.
- Give each test a concise explanation of what it checks, why that matters and
  how it checks it. Use a short docstring or focused comments rather than
  repeating the same explanation in both.

## Defects and verification

- Confirm suspected defects before recording them. Use narrowly scoped strict
  expected failures with explicit reasons for existing defects that this PR
  must not fix. Keep unaffected assertions in passing tests.
- Add no dependencies or options without a concrete need.
- Run the focused tests while developing, then the complete suite and appropriate
  lint/format checks. Repeat checks when changes or unresolved failures justify
  it; documentation-only changes do not require rerunning numerical tests.
- Summarize the behaviors covered, cases removed or consolidated, verification
  results, known defects and checks not performed. Do not claim publication or
  verification that did not happen.
