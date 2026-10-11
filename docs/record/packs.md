# Result packs — what each upload held

> **Kind:** record — which runs, audits and logs each results pack carried from the
> GPU machine to the analysis machine, results_2 to results_74. **Authoritative for:** what a
> citation such as "results_74" in the journey or the results record refers to. Closed on
> 8 October 2026, when the packs ended (journey §33.66).

Until 8 October 2026 the analysis ran on a separate machine, and results reached it as
zipped packs — by hand at first, then from `pack_results.py`, a retired script. When
the analysis moved to the GPU machine itself, every file in the 72 packs
still on disk (1,451) was compared with its copy in the project by SHA-256. None was
missing: 1,400 were byte-identical, 26 newer in the project —
run records from the first packs, later given the provenance fields of
`scripts/ops/backfill_results.py` (21); two runs re-run under the same id and packed again in
results_32; three logs packed while their chain was still running — and the
25 manifests held only what each run's own record holds. The packs were then moved to
the trash. A run's label, seed and numbers are in `results/<id>/`, or in
`results_invalid/<rule>/<id>/` if it was quarantined.

The file `results_59.zip` holds the pack numbered 60 (journey §33.34); no other file of
either number was kept, and there was no results_1. *Packed* is the
manifest's time where the pack had one, otherwise (marked \*) the zip file's time; UTC.

| pack | packed | runs | audits | logs | run ids |
|---|---|---|---|---|---|
| results_2 | 2026-07-28 20:11* | 3 | 0 | — | 03b9423bfe83 0e7cd8838192 735559832ec1 |
| results_3 | 2026-07-28 22:52* | 2 | 0 | — | 11e9fa7d35b3 51d794b3f24e |
| results_4 | 2026-07-29 01:25* | 2 | 0 | — | 88b6afa1e676 ccd320c0a0a0 |
| results_5 | 2026-08-01 03:17* | 3 | 0 | — | 873a52464643 96ebe413e6bf a7175857fdf6 |
| results_6 | 2026-08-07 03:08* | 2 | 0 | — | 55c394dba2e3 c69345978d4b |
| results_7 | 2026-08-10 14:34* | 3 | 0 | — | 1e57b24ed4f9 34a9e9b342da 56239ccfaf84 |
| results_8 | 2026-08-11 06:10* | 3 | 0 | — | 1fb14ea6f45d 3860406fd2c1 eed5d2a11a8a |
| results_9 | 2026-08-11 11:06* | 2 | 0 | — | a12b91da4049 e6b206f9a34a |
| results_10 | 2026-09-02 01:03* | 1 | 0 | — | 2d7b91872943 |
| results_11 | 2026-09-02 09:02* | 3 | 0 | — | 4e65750bc990 58d2637e8f69 c564d066df6b |
| results_12 | 2026-09-03 14:04* | 4 | 0 | — | 389a99d329ef 96e50812dedb b950c0b5ff55 f97f597841fa |
| results_13 | 2026-09-03 17:33* | 1 | 0 | — | d2af7d9ffe75 |
| results_14 | 2026-09-08 14:26* | 2 | 0 | — | ad6ea7603bfe e8cb4c7cc9c5 |
| results_15 | 2026-09-08 21:09* | 2 | 0 | — | a5ec220b62e3 fb35ef8a3634 |
| results_16 | 2026-09-09 08:57* | 1 | 0 | — | 4429c0884116 |
| results_17 | 2026-09-09 20:40* | 2 | 0 | — | 6302e2a46873 c1e0ab32e7fa |
| results_18 | 2026-09-10 00:49* | 2 | 0 | — | 0ea16fbc3b1f db9a289731bd |
| results_19 | 2026-09-10 07:39* | 12 | 0 | — | 14ee54560ab3 1bdf89e968cb 34a9b0270e95 3fb603cd87cb 58908b45bddc 65bbf11916c0 79c3688346a7 7ab2ab933b15 cfbb4ce93538 e3533a3b56c8 f349ff54f4ca ff4879ddfcd5 |
| results_20 | 2026-09-10 09:37* | 4 | 0 | — | 032174468a49 4b6c5a0ff5ae 6a9a245daf8a ed8b59c97675 |
| results_21 | 2026-09-12 00:51* | 4 | 0 | — | 033a4ad3064e 35e350b4c825 537b67213656 e3802f3aa50a |
| results_22 | 2026-09-20 01:59* | 5 | 0 | — | 0745320c7fe6 3d21c31c2544 67346ed11499 7a48ff0d17d4 b35ca4a7483a |
| results_23 | 2026-09-20 04:43* | 2 | 0 | — | 1bcbac60f6b5 979a7d1f5878 |
| results_24 | 2026-09-21 10:50* | 2 | 0 | — | 4de10670e169 ebf6ece5280a |
| results_25 | 2026-09-22 01:10* | 6 | 0 | — | 283aa0af77b6 3a3a4b2d927a 6d746b6333ca a3b5760012fd c56857a7f459 e556b64ef6db |
| results_26 | 2026-09-22 09:39* | 5 | 0 | — | 0745320c7fe6 3a3a4b2d927a 6d746b6333ca b35ca4a7483a c56857a7f459 |
| results_27 | 2026-09-22 11:06* | 1 | 0 | — | 956d58713c71 |
| results_28 | 2026-09-23 03:54* | 1 | 0 | — | a0abed930f58 |
| results_29 | 2026-09-23 07:20* | 1 | 0 | — | a0abed930f58 |
| results_30 | 2026-09-23 09:44* | 1 | 0 | — | 122cf133747d |
| results_31 | 2026-09-23 10:48* | 1 | 0 | — | 122cf133747d |
| results_32 | 2026-09-23 15:32* | 2 | 0 | — | 122cf133747d a0abed930f58 |
| results_33 | 2026-09-23 16:37* | 1 | 0 | — | 1b0847ff7660 |
| results_34 | 2026-09-25 07:29* | 2 | 0 | — | 5658981a9dc3 d5b4bc5ca05c |
| results_35 | 2026-09-25 09:26* | 2 | 0 | — | 05904594143e 648675b80ba0 |
| results_36 | 2026-09-25 19:33* | 2 | 0 | — | 54b18728cdfd 93556c318fcc |
| results_37 | 2026-09-25 21:58* | 2 | 0 | — | 1a6059aded6f e9e017b480c0 |
| results_38 | 2026-09-26 05:48* | 6 | 0 | — | 063318a88494 61dba34034ff 6388341bd574 c249334b1bdb f846a29a25a1 ff9efc229f5c |
| results_39 | 2026-09-26 08:04* | 2 | 0 | — | 70e02f2481d5 8ff1b23b16d0 |
| results_40 | 2026-09-26 22:37* | 8 | 26 | night_2026-09-26.log rebuild.log | 06b2a3bb429b 0b1f6c39cde6 2d8d8fbce97b 599e62772132 61156897cb7e 615cf6796212 a9f4a2434cfd b0dc64d5199a |
| results_41 | 2026-09-27 04:32* | 4 | 31 | — | 2a508f0432d6 85fc254b2c6d e189f96ecd25 ee684acb4e9a |
| results_42 | 2026-09-27 06:44* | 2 | 0 | — | 10b7adbeef86 c8fe0eb7cfe8 |
| results_43 | 2026-09-27 17:11* | 10 | 38 | night_2026-09-26.log night_2026-09-27.log rebuild.log | 0096ec5ce160 0e89c4564f5d 15cc32c37ce9 4b388cef56ed 8b7b7ac89cd8 8ba5756b2ccf 8ba64cd4a185 ac20641c273b d261f11e8c7e ffe63b490e43 |
| results_44 | 2026-09-27 21:31* | 4 | 43 | day_2026-09-28.log | 070773989bad 6b2a1889bbd6 6eede6137dc8 72b0e10a9502 |
| results_45 | 2026-09-28 01:03* | 2 | 0 | day_2026-09-28_additive.log | 3ad1db7f54bf 91d70215bdb3 |
| results_46 | 2026-09-28 04:17* | 2 | 43 | day_2026-09-28_control.log | 4b10c607b170 b971ec29693f |
| results_47 | 2026-09-28 07:41* | 3 | 43 | day_2026-09-28_control_s123.log | 5cd594bc376b d142e261f17d eddc9b47b40b |
| results_48 | 2026-09-28 22:10* | 13 | 46 | night_2026-09-28.log | 0746729523aa 1d75b444499f 4ee36311495c 645e5ef6542d 7052673db23b 7a46ca7f0ba9 875da4db22b8 937c0c16d504 98c578441d36 bba7b753f381 be73f14b107d dc748564cf70 fbbfb27db9ac |
| results_49 | 2026-09-28 22:48 | 0 | 3 | day_U0.log day_U1.log | — |
| results_50 | 2026-09-29 14:45 | 12 | 7 | day_U1b.log day_U1c.log night_2026-09-29.log night_2026-09-29_cpu.log | 11db800ede4d 2401ad4ac031 3ce1ac5acaaf 3d50d00486ad 6a878684e5f3 94293005c441 a871dbf91d50 abbcd92435d7 b1515377c268 bac159440880 bead5a1a1c56 ff47e21dad1a |
| results_51 | 2026-09-29 20:44 | 4 | 2 | day_D1.log day_D2.log | 0c7b5b1a0a9a 45dd7cb82ee3 9e630d686ea7 be8d53d3d692 |
| results_52 | 2026-09-30 01:02 | 4 | 0 | day_D3.log | 0ecad3b25755 60230f32f44d c62fbb6796c8 d280259a2e73 |
| results_53 | 2026-09-30 05:15 | 4 | 0 | day_D4.log | 872d42176726 9373d5eadb8a cc9249c7c4c3 f861b4d074f0 |
| results_54 | 2026-09-30 21:50 | 14 | 14 | night_2026-09-30.log night_2026-10-01.log night_2026-10-01_cpu.log | 575cbf7ff922 5fc95be4107a 712e9ef58f03 7158ce032bc0 7da5336046e7 8aa452ee93a4 8fbb7117749c a1736ea31b26 a7cccac36e94 ab9aca6d45cb cdc32b5a7b8d ce8884f45dbd dbfe3b1319f3 e496bac53fbb |
| results_55 | 2026-10-01 01:55 | 17 | 0 | night_2026-10-01.log | 0460a0982806 1547aa755e68 15b594460d95 26a549302a64 2bac1ea5e44d 3428c7be3f1d 39b07d927186 3e311e17cd67 4999fa96c493 74e59212de95 8034e2a46c53 93924df20400 cdc8f0a21b5b d3856a8fd692 e35ee3e99406 f36bc537955a fb6ac113262b |
| results_56 | 2026-10-01 09:48 | 33 | 11 | day_2026-10-01.log day_2026-10-01_A.log day_2026-10-01_B.log night_2026-10-01.log | 170a607c291d 193641468331 1f9cfb6b7a7a 261ea74b7c8b 29ec09d6d1c1 2ddf625019ef 4e2c79666d01 52376e28a71c 548b27d7f3fa 64b1463ead34 7176e851c097 72a166890758 73dfc038c0c7 79c12acf061e 87041e3a147b 894a98d84fa9 93283cc38857 9451c9db60f3 9a62a576fcd3 a5bc3063b85d a9877d77d9f4 b1e276db1cee b709519fc98a bbd5d2cfa20b bea766e07ac5 c5c6d777d2e4 c5da7b2c8311 cbabf4bf8725 cc42fc34e372 d3d204b40323 e4f2e0ec04f4 e956e170e7ad ee2850b59b7d |
| results_57 | 2026-10-01 12:00 | 4 | 0 | day_E1.log | 18d5ab19b68c 75d897725533 81d554621467 f3e4bf3c5d60 |
| results_58 | 2026-10-01 17:23 | 4 | 0 | day_A1.log | 0d863918f748 3e7d5e2e10b8 a1ca381fcca2 c02a49956d0f |
| results_59.zip, holding results_60 | 2026-10-01 23:52 | 4 | 0 | day_B1.log | 203b152df0de 4b8bdbc877f6 7a89912d99d0 d70b7dd27eac |
| results_61 | 2026-10-02 02:32 | 4 | 0 | day_C1.log | 3ab2a86d6396 525fe7956a2b 9fb684e66762 ff662a326eb6 |
| results_62 | 2026-10-02 18:05 | 56 | 2 | day_2026-10-02.log night_2026-10-02.log | 056044e9c3ca 08b87d14225a 14ff7195e84f 1a5304ae6579 21e862739c78 230cfe1a837f 23abda6bdc1e 24e08c01ccf2 2930118ce06d 29ebc5485c18 2a4ad71bfbfa 2cb626396c72 2f3f7fce9574 34ae45290f18 41fa495050d6 438f86ec47aa 4472aed3f78f 48103d34e324 4c06a10715c9 564c16dfa5c1 5e216409c566 61a08224fda4 6651e174051e 6e532335dd88 6ecaedc5da50 781cefc34270 78a986e877b1 7b635f39fd9e 82561f48b89b 8436e33083bb 869cc75b3095 8843dd6f0613 8b59ab151ec4 8e676e19faeb 96c4b1d4591e 97381f39b597 a2c6a6f13341 a423f715fa7c a51402d7535d a76c80832178 a796bef834d2 b7ba66d47a18 b8adf779ca7c b9e027dbe0ce bbfc6734e297 c3d0a3646baa c41d76dfb57f c9174ba30a01 c94803a6769d e13fc3edc3a6 e81b09fd5efd e86438a4402f e97ff509013a ec74fe64f743 f45247a40dff fb2139b344a2 |
| results_63 | 2026-10-03 02:52 | 2 | 0 | day_D.log | 7ab5b53eb521 cc10ec17d780 |
| results_64 | 2026-10-03 03:31 | 2 | 0 | day_D.log | b883601b7067 d2d886d98a54 |
| results_65 | 2026-10-03 18:50 | 34 | 1 | night_2026-10-03.log | 0f70eb8f4ed8 2a4088f908db 30e386181a2f 35cb6740af0a 397499a6096a 3c307b8589b0 44bc0e944578 575d809c7dd3 5a534474378b 691398c9e380 75442a0804d7 78a4f20011a4 78db298d34de 7c1c4832beb7 7e3d28edbe98 7fab1e01c1cd 82d7953cc676 8ce6e62a2766 8e687477ce09 8ee7c4c4d741 9205a2e15186 9901f0cab89a a5160049d059 a9869c299442 b0f8d49c03ad b96f2c35cc0b bc57c5f52505 d0db5b5f0972 d2f5a947df03 d90666eef9ea d9e0852caf30 e16d149bcd68 e6fe9fef7833 f5a618f263d8 |
| results_66 | 2026-10-03 19:09 | 1 | 0 | day_G69.log | 12e3d62efc84 |
| results_67 | 2026-10-04 19:33 | 17 | 1 | night_2026-10-04.log | 03c166c3db8b 12abac4fccb4 1a3bcbc47d80 1ddf55664126 31a7a7a600a1 327999f0ca2a 37b708743579 3ba6c98275cb 52c5cde5c8b0 5c3ef62b6541 6fd1efbd4dc6 8868df44c3e4 89f6d96fd996 952488f48af3 ae7e45a283c3 e07d2860450d f6138a99841c |
| results_68 | 2026-10-05 09:57 | 11 | 2 | day_2026-10-05.log screen_2026-10-05.log | 143fd951212b 1f5e2b066cb3 1f97db14b9c0 22c808860633 3fd4204e7a66 4b369bf2009d 6c2b70565e34 7ac653ec234e 88962a6f248a ac481e38d060 d0c1cb275412 |
| results_69 | 2026-10-05 17:34 | 19 | 3 | long_2026-10-05.log | 070edf8c4a7d 0983327a9d40 3c2dbef6bcb2 40074efa357e 40392d4c89d5 44c232437f0f 4cc1fae5a093 6b737f1c88fc 7915d2464a88 8a4f2aa96c97 92b576209fed 964b4c8368e6 a187f7ca915b ab3ca7c82024 d3d5c8b65347 e755fa7f6ddd f0f87882ba2b f4d6e7029ae9 f737038d9979 |
| results_70 | 2026-10-06 11:28 | 25 | 5 | long_2026-10-06.log piece_rope_zero.log | 066c70d64460 08baffc18218 16a11dd9bf91 1c13908ff38a 21f787fc62cb 2912c2d86a6a 3d31be625672 3f4f27924912 4517f6981718 45b52da275b0 4700798c6649 4dcdfe1f0b79 565222d8b666 5fc78da8a676 741edb183a05 7fcc5cf46fbb 8a45d99a8019 a0053dae7389 bbf981b5309a c364538685a3 d63e4fad0b6a d8d1a5de6387 e17715273354 e5e880e939e3 f70868eda99f |
| results_71 | 2026-10-07 01:39 | 23 | 55 | night_2026-10-06.log night_2026-10-06_cpu.log | 07dd78de8832 1ad58cb1c549 24b3759248eb 289bdbeaa301 2f3219758706 342503e36ecd 4d91d057644e 62e8c3848bcf 6ef7f8764712 6f214dfa186f 75bf7ae41554 b60e2ee58522 bda52b145424 c58ba0662048 d3bdd954bfb9 d81e089f0452 da6c527a22e8 dd4dbfca19de e1e912f50286 e9f3f153ec20 f7631513ec4e fe24a14162b2 fe50426ea6ef |
| results_72 | 2026-10-07 02:27 | 0 | 6 | probe_previous_1007.log | — |
| results_73 | 2026-10-07 07:15 | 0 | 8 | prep_bpeend_uk_17k.log | — |
| results_74 | 2026-10-07 18:30 | 48 | 3 | long_2026-10-07.log | 0405cdce32c0 0409e515dfb3 07e0d01dc63c 0e26ca4f1a5d 3d0f89e2516e 471ba4d35896 4ea43e41a4ae 4f5f719a1103 50c2fcca12d5 5f296b0d8d9b 6774555fdce6 68d9fdb55c11 6dda8d847d8a 6fadf8b09dd9 75d81834c028 7c5c22128e1a 8468c28dc6be 880c361eec57 8812d74b6691 8c9007651234 8de38381d30e 9b80b67971df 9bef22495fb6 a369ea5bc2a7 a5bdbcf3f1fd a65f3bf0ad1d ad927855ba92 adad9d549369 bb9186bb4b9c bc8af89c7937 be5940ecca81 bea4ae10a12e c8b4ec74bce6 cb2029bca422 cc7416e8e99e cd499b2510a3 ce0e29de6e1b cfc255a98208 da283a3f5d8b db008d746a18 de5adf9e437c e000c1ef94ed e622a134e481 e6998288f97e ea5603cbda17 ec2e283e41a4 ed3afe0e7a24 fdd6184c658e |
