# 首先，你需要安装并加载ggplot2包
# install.packages("ggplot2")
library(ggplot2)

# 由于数据是手动输入的，我们将其整理成数据框格式
data <- data.frame(
  Value = c(1.92E-11, 9.28E-12, 1.19E-11, 1.08E-11, 3.30E-11, 2.27E-11, 2.14E-11, 1.33E-11, 1.02E-11, 9.66E-12, 9.50E-12, 1.72E-11, 1.27E-11, 1.22E-11, 1.20E-11, 1.19E-11, 2.22E-11, 1.50E-11, 1.07E-11, 1.02E-11),
  Group = rep(c("Others", "Others", "Immune", "Immune", "Cellular metabolism and transport", "Cellular metabolism and transport", "Cellular metabolism and transport", "Cellular metabolism and transport", "Cellular metabolism and transport", "Cellular metabolism and transport", "Cellular metabolism and transport", "Cellular process and function", "Cellular process and function", "Cellular process and function", "Cellular process and function", "Cellular process and function", "Signal transduction", "Signal transduction", "Signal transduction", "Signal transduction"), times = 1),
  Category = c("KEGG_GLUTATHIONE_METABOLISM", "REACTOME_DOWNREGULATION_OF_ERBB4_SIGNALING", "WP_CELLS_AND_MOLECULES_INVOLVED_IN_LOCAL_ACUTE_INFLAMMATORY_RESPONSE", "WP_EBOLA_VIRUS_INFECTION_IN_HOST", "REACTOME_PYRUVATE_METABOLISM", "KEGG_FOLATE_BIOSYNTHESIS", "REACTOME_CITRIC_ACID_CYCLE_TCA_CYCLE", "REACTOME_LDL_REMODELING", "REACTOME_PEROXISOMAL_LIPID_METABOLISM", "WP_VITAMIN_B12_DISORDERS", "REACTOME_PLASMA_LIPOPROTEIN_REMODELING", "REACTOME_PROTEIN_REPAIR", "REACTOME_ATF6_ATF6_ALPHA_ACTIVATES_CHAPERONE_GENES", "WP_TCA_CYCLE_IN_SENESCENCE", "REACTOME_G2_PHASE", "BIOMCARTA_LAIR_PATHWAY", "REACTOME_NR1H2_NR1H3_REGULATE_GENE_EXPRESSION_TO_LIMIT_CHOLESTEROL_UPTAKE", "REACTOME_REGULATION_OF_FZD_BY_UBIQUITINATION", "WP_PPARALPHA_PATHWAY", "WP_BMP2WNT4FOXO1_PATHWAY_IN_PRIMARY_ENDOMETRIAL_STROMAL_CELL_DIFFERENTIATION")
)

# 使用ggplot2绘制横向条形图
ggplot(data, aes(x = reorder(Category, Value), y = Value, fill = Group)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() + # 使条形图横向显示
  theme_classic() + # 添加简洁的主题
  labs(x = "Category", y = "Value", fill = "Group") + # 添加轴标签和图例标题
  theme(legend.position = "bottom") # 将图例放置在图形下方
