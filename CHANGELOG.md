# Changelog

## [0.2.0](https://github.com/BlueBrain/data-validation-framework/compare/0.1.1..0.2.0)

> 20 December 2021

- Add a mixin to make the task skippable [#7](https://github.com/BlueBrain/data-validation-framework/pull/7)
- Setup Codecov [#6](https://github.com/BlueBrain/data-validation-framework/pull/6)

## [0.1.1](https://github.com/BlueBrain/data-validation-framework/compare/0.1.0..0.1.1)

> 15 December 2021

- Fix typos in README [#4](https://github.com/BlueBrain/data-validation-framework/pull/4)

## [0.1.0](https://github.com/BlueBrain/data-validation-framework/compare/0.0.9..0.1.0)

> 15 December 2021

- Link fixed in README.md [#2](https://github.com/BlueBrain/data-validation-framework/pull/2)
- Change license and open the sources [#1](https://github.com/BlueBrain/data-validation-framework/pull/1)
- Add contribution doc (Adrien Berchet - [ebfe669](https://github.com/BlueBrain/data-validation-framework/commit/ebfe6698ae270e0873ddde8d1e0157a02cc50f2e))
- Drop support of Python 3.6 and 3.7 (Adrien Berchet - [5bef8b0](https://github.com/BlueBrain/data-validation-framework/commit/5bef8b00cacb4e1c4210e8444fde3ef5ea95b7be))
- Fix auto-changelog (Adrien Berchet - [5529558](https://github.com/BlueBrain/data-validation-framework/commit/55295589e374b9ef76730905704593c4ffefe68b))
- Use luigi_tools.parameter.OptionalPathParameter for the dataset path (Adrien Berchet - [eeb1f84](https://github.com/BlueBrain/data-validation-framework/commit/eeb1f847016f2e1aa678c51e5a768a70b18fe36c))

## [0.0.9](https://github.com/BlueBrain/data-validation-framework/compare/0.0.8..0.0.9)

> 18 October 2021

- Setup auto-changelog and add authors and license (Adrien Berchet - [590c5ad](https://github.com/BlueBrain/data-validation-framework/commit/590c5ad92a285f088eb64bcdcbbb59a524c266b3))
- Add a warning when TagResultOutputMixin is used alongside RerunMixin (Adrien Berchet - [f80f6da](https://github.com/BlueBrain/data-validation-framework/commit/f80f6dab1d44134a39b1dd74f17affaa65927bb6))
- Export test and coverage reports to GitLab (Adrien Berchet - [c1f3b29](https://github.com/BlueBrain/data-validation-framework/commit/c1f3b29f281ce008a2fbd9f0a6444a644e0ce817))

## [0.0.8](https://github.com/BlueBrain/data-validation-framework/compare/0.0.7..0.0.8)

> 13 July 2021

- Add extra_requires feature to use regular luigi tasks in validation workflows (Alexis Arnaudon - [35518e7](https://github.com/BlueBrain/data-validation-framework/commit/35518e7790b3145ba354098ef9e113e4fb817ee4))
- Use custom image in CI jobs (Adrien Berchet - [215a1b8](https://github.com/BlueBrain/data-validation-framework/commit/215a1b8fa7d887d70c561dc0fac23ce1de4e6dd0))

## [0.0.7](https://github.com/BlueBrain/data-validation-framework/compare/0.0.6..0.0.7)

> 28 June 2021

- Remove some warnings for exceptions in nested tasks (Adrien Berchet - [4e754ee](https://github.com/BlueBrain/data-validation-framework/commit/4e754ee0356b6643ee16ba6dafc4cd3ec9b099f9))

## [0.0.6](https://github.com/BlueBrain/data-validation-framework/compare/0.0.5..0.0.6)

> 28 June 2021

- Add todo extension to Sphinx and fix ret_code handling for SetValidationTask (Adrien Berchet - [529200b](https://github.com/BlueBrain/data-validation-framework/commit/529200bc6f7091945908dbe706c82f7279037021))
- Migration from gerrit to github (Adrien Berchet - [80dc073](https://github.com/BlueBrain/data-validation-framework/commit/80dc0737c4535c335da60d27209e5eff616eccb2))
- Add auto-release job in CI (Adrien Berchet - [2317324](https://github.com/BlueBrain/data-validation-framework/commit/2317324341fedf020d01faedca349bc48819f9ff))

## [0.0.5](https://github.com/BlueBrain/data-validation-framework/compare/0.0.4..0.0.5)

> 19 April 2021

- Improve warning mechanism and add option to not capture stdout in validation function (Adrien Berchet - [ce63be9](https://github.com/BlueBrain/data-validation-framework/commit/ce63be90a5aa4dfc1f628461fbfcada721bb786b))
- Fix: use __specifications__ instead of __doc__ to generate reports (Adrien Berchet - [0fc3f97](https://github.com/BlueBrain/data-validation-framework/commit/0fc3f97238faa8d0b6f8ff2f5797ef4809a4ddf5))

## [0.0.4](https://github.com/BlueBrain/data-validation-framework/compare/0.0.3..0.0.4)

> 22 March 2021

- Add multiprocessing feature to apply_to_df (Adrien Berchet - [c28a0c1](https://github.com/BlueBrain/data-validation-framework/commit/c28a0c1500c22b5c32d3b150986cf283169f1281))
- Update parameter propagation (Adrien Berchet - [ed79ae5](https://github.com/BlueBrain/data-validation-framework/commit/ed79ae5807fa97a366e994a3191bd339272a2270))
- Fix tqdm interference with prints (Adrien Berchet - [7ecf9c8](https://github.com/BlueBrain/data-validation-framework/commit/7ecf9c88d18cd24f925cb54c2d0b2aa70a557d14))
- Fix progress bar with multiprocessing computation (Adrien Berchet - [c76bf5f](https://github.com/BlueBrain/data-validation-framework/commit/c76bf5f9d9f7103d608b8e8b15e1f02ceb4988e3))
- External validation functions no more need to be declared as staticfunction (Adrien Berchet - [ed171c0](https://github.com/BlueBrain/data-validation-framework/commit/ed171c064c341f9744f52b4d9ff04e917636d048))

## [0.0.3](https://github.com/BlueBrain/data-validation-framework/compare/0.0.2..0.0.3)

> 8 March 2021

- Improve specification generation (Adrien Berchet - [dc79c30](https://github.com/BlueBrain/data-validation-framework/commit/dc79c30baf0fbf8661b162158d71e5947b8e0551))

## 0.0.2

> 3 March 2021

- First commit (Adrien Berchet - [43a8cee](https://github.com/BlueBrain/data-validation-framework/commit/43a8ceeb540f76e282c04f449bd4b149e2ab4027))
- Remove test_version (Adrien Berchet - [5db162e](https://github.com/BlueBrain/data-validation-framework/commit/5db162e1a1ee1a187f4f234761c8c19f6f33108e))
- Fix versioning (Adrien Berchet - [0755e85](https://github.com/BlueBrain/data-validation-framework/commit/0755e85bd5e3d31779a7a55a7fd28c1cf752f184))
- Initial empty repository (Dries Verachtert - [890f91e](https://github.com/BlueBrain/data-validation-framework/commit/890f91ee5af155e6fea2f79e118f153291fa7975))
