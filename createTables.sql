
CREATE TABLE `comments` (
  `post_shortcode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '',
  `comment_username` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '',
  `comment_text` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `comment_timestamp` datetime NOT NULL DEFAULT '0000-00-00 00:00:00',
  `comment_vector` blob,
  PRIMARY KEY (`post_shortcode`,`comment_username`,`comment_timestamp`),
  SHARD KEY `__SHARDKEY` (`post_shortcode`,`comment_username`,`comment_timestamp`),
  SORT KEY `__UNORDERED` ()
) AUTOSTATS_CARDINALITY_MODE=INCREMENTAL AUTOSTATS_HISTOGRAM_MODE=CREATE AUTOSTATS_SAMPLING=ON SQL_MODE='STRICT_ALL_TABLES,NO_AUTO_CREATE_USER';

CREATE TABLE `media` (
  `post_shortcode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `media_file` longblob,
  `media_type` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  SORT KEY `__UNORDERED` ()
  , SHARD KEY () 
) AUTOSTATS_CARDINALITY_MODE=INCREMENTAL AUTOSTATS_HISTOGRAM_MODE=CREATE AUTOSTATS_SAMPLING=ON SQL_MODE='STRICT_ALL_TABLES,NO_AUTO_CREATE_USER';

CREATE TABLE `posts` (
  `topic` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `post_shortcode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '',
  `post_url` varchar(1255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `post_timestamp` datetime DEFAULT NULL,
  `meta_text` varchar(1255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  PRIMARY KEY (`post_shortcode`),
  SHARD KEY `__SHARDKEY` (`post_shortcode`),
  SORT KEY `__UNORDERED` ()
) AUTOSTATS_CARDINALITY_MODE=INCREMENTAL AUTOSTATS_HISTOGRAM_MODE=CREATE AUTOSTATS_SAMPLING=ON SQL_MODE='STRICT_ALL_TABLES,NO_AUTO_CREATE_USER';
