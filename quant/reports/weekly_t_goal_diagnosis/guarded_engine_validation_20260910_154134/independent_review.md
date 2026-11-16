# 独立只读复核

inspect_entry审阅新旧差异及原calculate_quote_fill，确认quantity=0/used=0不要求虚拟反向容量，不变更现金/旧股/预留/used；取消在任何fill_book之前；后续恢复行情不参与；未发现阻断缺陷。关闭状态只增加审计表。

补充口径：own_fillable_shares是本方向quote/容量可填量，尚未经过现金、准备金和pending上限，不代表原本必然成交数量。表粒度是到达报价复核请求，不能作为已成交T组分母。

此复核未修改任何源码；生产48旧源与新的guarded源均维持归档SHA。
