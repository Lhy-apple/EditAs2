/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Wed Jul 12 06:00:11 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

import static org.evosuite.shaded.org.mockito.Mockito.*;
@EvoSuiteClassExclude
public class ObjectIdReferenceProperty_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "com.fasterxml.jackson.databind.deser.impl.ObjectIdReferenceProperty"; 
    org.evosuite.runtime.GuiSupport.initialize(); 
    org.evosuite.runtime.RuntimeSettings.maxNumberOfThreads = 100; 
    org.evosuite.runtime.RuntimeSettings.maxNumberOfIterationsPerLoop = 10000; 
    org.evosuite.runtime.RuntimeSettings.mockSystemIn = true; 
    org.evosuite.runtime.RuntimeSettings.sandboxMode = org.evosuite.runtime.sandbox.Sandbox.SandboxMode.RECOMMENDED; 
    org.evosuite.runtime.sandbox.Sandbox.initializeSecurityManagerForSUT(); 
    org.evosuite.runtime.classhandling.JDKClassResetter.init();
    setSystemProperties();
    initializeClasses();
    org.evosuite.runtime.Runtime.getInstance().resetRuntime(); 
    try { initMocksToAvoidTimeoutsInTheTests(); } catch(ClassNotFoundException e) {} 
  } 

  @AfterClass 
  public static void clearEvoSuiteFramework(){ 
    Sandbox.resetDefaultSecurityManager(); 
    java.lang.System.setProperties((java.util.Properties) defaultProperties.clone()); 
  } 

  @Before 
  public void initTestCase(){ 
    threadStopper.storeCurrentThreads();
    threadStopper.startRecordingTime();
    org.evosuite.runtime.jvm.ShutdownHookHandler.getInstance().initHandler(); 
    org.evosuite.runtime.sandbox.Sandbox.goingToExecuteSUTCode(); 
    setSystemProperties(); 
    org.evosuite.runtime.GuiSupport.setHeadless(); 
    org.evosuite.runtime.Runtime.getInstance().resetRuntime(); 
    org.evosuite.runtime.agent.InstrumentingAgent.activate(); 
  } 

  @After 
  public void doneWithTestCase(){ 
    threadStopper.killAndJoinClientThreads();
    org.evosuite.runtime.jvm.ShutdownHookHandler.getInstance().safeExecuteAddedHooks(); 
    org.evosuite.runtime.classhandling.JDKClassResetter.reset(); 
    resetClasses(); 
    org.evosuite.runtime.sandbox.Sandbox.doneWithExecutingSUTCode(); 
    org.evosuite.runtime.agent.InstrumentingAgent.deactivate(); 
    org.evosuite.runtime.GuiSupport.restoreHeadlessMode(); 
  } 

  public static void setSystemProperties() {
 
    java.lang.System.setProperties((java.util.Properties) defaultProperties.clone()); 
    java.lang.System.setProperty("user.dir", "/data/swf/zenodo_replication_package_new"); 
    java.lang.System.setProperty("java.io.tmpdir", "/tmp"); 
  }

  private static void initializeClasses() {
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(ObjectIdReferenceProperty_ESTest_scaffolding.class.getClassLoader() ,
      "com.fasterxml.jackson.core.JsonLocation",
      "com.fasterxml.jackson.databind.JsonMappingException$Reference",
      "com.fasterxml.jackson.databind.deser.impl.ReadableObjectId",
      "com.fasterxml.jackson.core.JsonGenerator",
      "com.fasterxml.jackson.core.SerializableString",
      "com.fasterxml.jackson.databind.jsontype.TypeDeserializer",
      "com.fasterxml.jackson.core.Versioned",
      "com.fasterxml.jackson.databind.util.AccessPattern",
      "com.fasterxml.jackson.databind.DatabindContext",
      "com.fasterxml.jackson.databind.DeserializationFeature",
      "com.fasterxml.jackson.databind.introspect.BasicBeanDescription",
      "com.fasterxml.jackson.databind.deser.std.StdDeserializer",
      "com.fasterxml.jackson.databind.MapperFeature",
      "com.fasterxml.jackson.databind.DeserializationConfig",
      "com.fasterxml.jackson.databind.deser.ResolvableDeserializer",
      "com.fasterxml.jackson.databind.deser.UnresolvedForwardReference",
      "com.fasterxml.jackson.databind.cfg.MapperConfigBase",
      "com.fasterxml.jackson.databind.introspect.AnnotatedMember",
      "com.fasterxml.jackson.databind.util.ClassUtil",
      "com.fasterxml.jackson.databind.BeanDescription",
      "com.fasterxml.jackson.databind.SerializerProvider",
      "com.fasterxml.jackson.databind.JsonDeserializer",
      "com.fasterxml.jackson.databind.introspect.ConcreteBeanPropertyBase",
      "com.fasterxml.jackson.core.JsonParseException",
      "com.fasterxml.jackson.databind.introspect.ObjectIdInfo",
      "com.fasterxml.jackson.databind.deser.impl.ObjectIdReader",
      "com.fasterxml.jackson.databind.util.Named",
      "com.fasterxml.jackson.databind.deser.ContextualDeserializer",
      "com.fasterxml.jackson.databind.DeserializationContext",
      "com.fasterxml.jackson.core.JsonParser",
      "com.fasterxml.jackson.databind.PropertyName",
      "com.fasterxml.jackson.databind.BeanProperty",
      "com.fasterxml.jackson.databind.deser.NullValueProvider",
      "com.fasterxml.jackson.core.JsonProcessingException",
      "com.fasterxml.jackson.databind.deser.impl.ReadableObjectId$Referring",
      "com.fasterxml.jackson.databind.deser.std.StdDelegatingDeserializer",
      "com.fasterxml.jackson.databind.introspect.Annotated",
      "com.fasterxml.jackson.databind.cfg.ConfigFeature",
      "com.fasterxml.jackson.databind.deser.impl.ObjectIdReferenceProperty$PropertyReferring",
      "com.fasterxml.jackson.databind.util.NameTransformer",
      "com.fasterxml.jackson.databind.deser.impl.FailingDeserializer",
      "com.fasterxml.jackson.databind.util.ClassUtil$Ctor",
      "com.fasterxml.jackson.databind.cfg.MapperConfig",
      "com.fasterxml.jackson.databind.deser.impl.ObjectIdReferenceProperty",
      "com.fasterxml.jackson.databind.JsonMappingException",
      "com.fasterxml.jackson.databind.introspect.ClassIntrospector$MixInResolver",
      "com.fasterxml.jackson.databind.deser.SettableBeanProperty"
    );
  } 
  private static void initMocksToAvoidTimeoutsInTheTests() throws ClassNotFoundException { 
    mock(Class.forName("com.fasterxml.jackson.databind.JsonDeserializer", false, ObjectIdReferenceProperty_ESTest_scaffolding.class.getClassLoader()));
  }

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(ObjectIdReferenceProperty_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "com.fasterxml.jackson.databind.introspect.ConcreteBeanPropertyBase",
      "com.fasterxml.jackson.databind.JsonDeserializer",
      "com.fasterxml.jackson.databind.DeserializationFeature",
      "com.fasterxml.jackson.databind.deser.std.StdDeserializer",
      "com.fasterxml.jackson.databind.deser.impl.FailingDeserializer",
      "com.fasterxml.jackson.databind.deser.SettableBeanProperty",
      "com.fasterxml.jackson.databind.deser.impl.ObjectIdReferenceProperty",
      "com.fasterxml.jackson.databind.deser.impl.ReadableObjectId$Referring",
      "com.fasterxml.jackson.databind.deser.impl.ObjectIdReferenceProperty$PropertyReferring",
      "com.fasterxml.jackson.annotation.JsonFormat$Shape",
      "com.fasterxml.jackson.annotation.JsonFormat$Features",
      "com.fasterxml.jackson.annotation.JsonFormat$Value",
      "com.fasterxml.jackson.annotation.JsonInclude$Include",
      "com.fasterxml.jackson.annotation.JsonInclude$Value",
      "com.fasterxml.jackson.databind.BeanProperty",
      "com.fasterxml.jackson.databind.BeanDescription",
      "com.fasterxml.jackson.databind.introspect.BasicBeanDescription",
      "com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector",
      "com.fasterxml.jackson.databind.deser.DeserializerFactory",
      "com.fasterxml.jackson.databind.util.ClassUtil",
      "com.fasterxml.jackson.databind.PropertyName",
      "com.fasterxml.jackson.databind.deser.BasicDeserializerFactory",
      "com.fasterxml.jackson.databind.deser.std.StdKeyDeserializers",
      "com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig",
      "com.fasterxml.jackson.databind.deser.BeanDeserializerFactory",
      "com.fasterxml.jackson.databind.DatabindContext",
      "com.fasterxml.jackson.databind.DeserializationContext",
      "com.fasterxml.jackson.databind.deser.DefaultDeserializationContext",
      "com.fasterxml.jackson.databind.deser.DefaultDeserializationContext$Impl",
      "com.fasterxml.jackson.databind.deser.DeserializerCache",
      "com.fasterxml.jackson.databind.deser.BeanDeserializerBuilder",
      "com.fasterxml.jackson.databind.deser.impl.BeanPropertyMap",
      "com.fasterxml.jackson.databind.deser.BeanDeserializerBase",
      "com.fasterxml.jackson.databind.deser.BeanDeserializer",
      "com.fasterxml.jackson.core.util.InternCache",
      "com.fasterxml.jackson.core.filter.TokenFilter",
      "com.fasterxml.jackson.core.JsonStreamContext",
      "com.fasterxml.jackson.core.filter.TokenFilterContext",
      "com.fasterxml.jackson.core.JsonLocation",
      "com.fasterxml.jackson.annotation.ObjectIdGenerator$IdKey",
      "com.fasterxml.jackson.databind.deser.AbstractDeserializer",
      "com.fasterxml.jackson.databind.introspect.ObjectIdInfo",
      "com.fasterxml.jackson.core.type.ResolvedType",
      "com.fasterxml.jackson.databind.JavaType",
      "com.fasterxml.jackson.databind.type.TypeBindings",
      "com.fasterxml.jackson.databind.type.TypeBase",
      "com.fasterxml.jackson.databind.type.PlaceholderForType",
      "com.fasterxml.jackson.databind.util.LRUMap",
      "com.fasterxml.jackson.databind.type.TypeParser",
      "com.fasterxml.jackson.databind.type.SimpleType",
      "com.fasterxml.jackson.databind.type.TypeFactory",
      "com.fasterxml.jackson.databind.type.ReferenceType",
      "com.fasterxml.jackson.databind.deser.std.StdScalarDeserializer",
      "com.fasterxml.jackson.databind.deser.std.FromStringDeserializer",
      "com.fasterxml.jackson.databind.ext.CoreXMLDeserializers$Std",
      "com.fasterxml.jackson.core.JsonProcessingException",
      "com.fasterxml.jackson.databind.JsonMappingException",
      "com.fasterxml.jackson.databind.deser.UnresolvedForwardReference",
      "com.fasterxml.jackson.databind.introspect.ClassIntrospector",
      "com.fasterxml.jackson.databind.introspect.AnnotationCollector$NoAnnotations",
      "com.fasterxml.jackson.databind.introspect.AnnotationCollector",
      "com.fasterxml.jackson.databind.introspect.AnnotatedClassResolver",
      "com.fasterxml.jackson.databind.introspect.Annotated",
      "com.fasterxml.jackson.databind.introspect.AnnotatedClass$Creators",
      "com.fasterxml.jackson.databind.introspect.AnnotatedClass",
      "com.fasterxml.jackson.databind.introspect.BasicClassIntrospector",
      "com.fasterxml.jackson.databind.type.CollectionLikeType",
      "com.fasterxml.jackson.databind.BeanProperty$Bogus",
      "com.fasterxml.jackson.databind.AnnotationIntrospector",
      "com.fasterxml.jackson.databind.introspect.NopAnnotationIntrospector$1",
      "com.fasterxml.jackson.databind.introspect.NopAnnotationIntrospector",
      "com.fasterxml.jackson.databind.PropertyNamingStrategy$PropertyNamingStrategyBase",
      "com.fasterxml.jackson.databind.PropertyNamingStrategy$SnakeCaseStrategy",
      "com.fasterxml.jackson.databind.PropertyNamingStrategy$UpperCamelCaseStrategy",
      "com.fasterxml.jackson.databind.PropertyNamingStrategy$LowerCaseStrategy",
      "com.fasterxml.jackson.databind.PropertyNamingStrategy$KebabCaseStrategy",
      "com.fasterxml.jackson.databind.PropertyNamingStrategy",
      "com.fasterxml.jackson.databind.ser.BeanSerializerBuilder",
      "com.fasterxml.jackson.annotation.ObjectIdGenerator",
      "com.fasterxml.jackson.annotation.SimpleObjectIdResolver",
      "com.fasterxml.jackson.databind.Module",
      "com.fasterxml.jackson.databind.module.SimpleModule",
      "com.fasterxml.jackson.core.Version",
      "com.fasterxml.jackson.databind.jsontype.NamedType",
      "com.fasterxml.jackson.databind.util.NameTransformer$NopTransformer",
      "com.fasterxml.jackson.databind.util.NameTransformer",
      "com.fasterxml.jackson.databind.type.ClassStack",
      "com.fasterxml.jackson.core.util.BufferRecycler",
      "com.fasterxml.jackson.databind.JsonMappingException$Reference",
      "com.fasterxml.jackson.core.util.ByteArrayBuilder",
      "com.fasterxml.jackson.core.io.IOContext",
      "com.fasterxml.jackson.core.io.SerializedString",
      "com.fasterxml.jackson.core.util.DefaultPrettyPrinter",
      "com.fasterxml.jackson.core.JsonFactory",
      "com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer",
      "com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer$TableInfo",
      "com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer",
      "com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer$TableInfo",
      "com.fasterxml.jackson.databind.JsonSerializer",
      "com.fasterxml.jackson.databind.ser.std.StdSerializer",
      "com.fasterxml.jackson.databind.ser.impl.FailingSerializer",
      "com.fasterxml.jackson.databind.ser.impl.UnknownSerializer",
      "com.fasterxml.jackson.databind.SerializerProvider",
      "com.fasterxml.jackson.databind.ser.DefaultSerializerProvider",
      "com.fasterxml.jackson.databind.ser.DefaultSerializerProvider$Impl",
      "com.fasterxml.jackson.databind.ser.std.NullSerializer",
      "com.fasterxml.jackson.databind.ser.SerializerCache",
      "com.fasterxml.jackson.core.TreeCodec",
      "com.fasterxml.jackson.core.ObjectCodec",
      "com.fasterxml.jackson.databind.ext.Java7SupportImpl",
      "com.fasterxml.jackson.databind.ext.Java7Support",
      "com.fasterxml.jackson.databind.introspect.JacksonAnnotationIntrospector",
      "com.fasterxml.jackson.databind.cfg.BaseSettings",
      "com.fasterxml.jackson.databind.util.StdDateFormat",
      "com.fasterxml.jackson.core.Base64Variant",
      "com.fasterxml.jackson.core.Base64Variants",
      "com.fasterxml.jackson.databind.ObjectMapper",
      "com.fasterxml.jackson.databind.jsontype.SubtypeResolver",
      "com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver",
      "com.fasterxml.jackson.databind.util.RootNameLookup",
      "com.fasterxml.jackson.databind.introspect.SimpleMixInResolver",
      "com.fasterxml.jackson.databind.cfg.ConfigOverrides",
      "com.fasterxml.jackson.annotation.JsonSetter$Value",
      "com.fasterxml.jackson.databind.introspect.VisibilityChecker$Std",
      "com.fasterxml.jackson.databind.cfg.MapperConfig",
      "com.fasterxml.jackson.databind.cfg.ConfigOverride",
      "com.fasterxml.jackson.databind.cfg.ConfigOverride$Empty",
      "com.fasterxml.jackson.databind.cfg.MapperConfigBase",
      "com.fasterxml.jackson.core.util.DefaultPrettyPrinter$NopIndenter",
      "com.fasterxml.jackson.core.util.DefaultPrettyPrinter$FixedSpaceIndenter",
      "com.fasterxml.jackson.core.util.DefaultIndenter",
      "com.fasterxml.jackson.core.util.Separators",
      "com.fasterxml.jackson.core.PrettyPrinter",
      "com.fasterxml.jackson.databind.SerializationConfig",
      "com.fasterxml.jackson.databind.cfg.ContextAttributes",
      "com.fasterxml.jackson.databind.cfg.ContextAttributes$Impl",
      "com.fasterxml.jackson.databind.DeserializationConfig",
      "com.fasterxml.jackson.databind.node.JsonNodeFactory",
      "com.fasterxml.jackson.databind.ser.SerializerFactory",
      "com.fasterxml.jackson.databind.ser.std.StdScalarSerializer",
      "com.fasterxml.jackson.databind.ser.std.StringSerializer",
      "com.fasterxml.jackson.databind.ser.std.ToStringSerializer",
      "com.fasterxml.jackson.databind.ser.std.NumberSerializers",
      "com.fasterxml.jackson.databind.ser.std.NumberSerializers$Base",
      "com.fasterxml.jackson.databind.ser.std.NumberSerializers$IntegerSerializer",
      "com.fasterxml.jackson.core.JsonParser$NumberType",
      "com.fasterxml.jackson.databind.ser.std.NumberSerializers$LongSerializer",
      "com.fasterxml.jackson.databind.ser.std.NumberSerializers$IntLikeSerializer",
      "com.fasterxml.jackson.databind.ser.std.NumberSerializers$ShortSerializer",
      "com.fasterxml.jackson.databind.ser.std.NumberSerializers$DoubleSerializer",
      "com.fasterxml.jackson.databind.ser.std.NumberSerializers$FloatSerializer",
      "com.fasterxml.jackson.databind.ser.std.BooleanSerializer",
      "com.fasterxml.jackson.databind.ser.std.NumberSerializer",
      "com.fasterxml.jackson.databind.ser.std.DateTimeSerializerBase",
      "com.fasterxml.jackson.databind.ser.std.CalendarSerializer",
      "com.fasterxml.jackson.databind.ser.std.DateSerializer",
      "com.fasterxml.jackson.databind.ser.std.StdJdkSerializers",
      "com.fasterxml.jackson.databind.ser.std.UUIDSerializer",
      "com.fasterxml.jackson.databind.ser.BasicSerializerFactory",
      "com.fasterxml.jackson.databind.cfg.SerializerFactoryConfig",
      "com.fasterxml.jackson.databind.ser.BeanSerializerFactory",
      "com.fasterxml.jackson.core.type.TypeReference",
      "com.fasterxml.jackson.databind.introspect.AnnotationIntrospectorPair",
      "com.fasterxml.jackson.databind.introspect.BeanPropertyDefinition",
      "com.fasterxml.jackson.databind.AnnotationIntrospector$ReferenceProperty",
      "com.fasterxml.jackson.databind.AnnotationIntrospector$ReferenceProperty$Type",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$8",
      "com.fasterxml.jackson.databind.MappingJsonFactory",
      "com.fasterxml.jackson.databind.ser.impl.ReadOnlyClassToSerializerMap",
      "com.fasterxml.jackson.databind.util.TypeKey",
      "com.fasterxml.jackson.databind.introspect.AnnotationCollector$EmptyCollector",
      "com.fasterxml.jackson.databind.util.ArrayIterator",
      "com.fasterxml.jackson.databind.introspect.CollectorBase",
      "com.fasterxml.jackson.databind.introspect.AnnotatedFieldCollector",
      "com.fasterxml.jackson.databind.introspect.TypeResolutionContext$Basic",
      "com.fasterxml.jackson.databind.introspect.AnnotatedFieldCollector$FieldBuilder",
      "com.fasterxml.jackson.databind.introspect.AnnotatedMember",
      "com.fasterxml.jackson.databind.introspect.AnnotatedField",
      "com.fasterxml.jackson.databind.introspect.AnnotationMap",
      "com.fasterxml.jackson.annotation.JsonAutoDetect$1",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$Linked",
      "com.fasterxml.jackson.databind.introspect.AnnotatedMethodCollector",
      "com.fasterxml.jackson.databind.introspect.MemberKey",
      "com.fasterxml.jackson.databind.introspect.AnnotatedMethodCollector$MethodBuilder",
      "com.fasterxml.jackson.databind.introspect.AnnotatedWithParams",
      "com.fasterxml.jackson.databind.introspect.AnnotatedMethod",
      "com.fasterxml.jackson.databind.introspect.AnnotatedMethodMap",
      "com.fasterxml.jackson.databind.util.BeanUtil",
      "com.fasterxml.jackson.databind.introspect.AnnotatedCreatorCollector",
      "com.fasterxml.jackson.databind.util.ClassUtil$Ctor",
      "com.fasterxml.jackson.databind.introspect.AnnotatedConstructor",
      "com.fasterxml.jackson.databind.introspect.AnnotatedParameter",
      "com.fasterxml.jackson.databind.type.ArrayType",
      "com.fasterxml.jackson.core.JsonGenerator",
      "com.fasterxml.jackson.core.base.GeneratorBase",
      "com.fasterxml.jackson.core.io.CharTypes",
      "com.fasterxml.jackson.core.json.JsonGeneratorImpl",
      "com.fasterxml.jackson.core.json.UTF8JsonGenerator",
      "com.fasterxml.jackson.core.json.JsonWriteContext",
      "com.fasterxml.jackson.core.util.VersionUtil",
      "com.fasterxml.jackson.databind.cfg.PackageVersion",
      "com.fasterxml.jackson.core.JsonParser",
      "com.fasterxml.jackson.core.util.JsonParserDelegate",
      "com.fasterxml.jackson.databind.module.SimpleKeyDeserializers",
      "com.fasterxml.jackson.databind.util.ArrayBuilders",
      "com.fasterxml.jackson.annotation.ObjectIdGenerators$Base",
      "com.fasterxml.jackson.annotation.ObjectIdGenerators$IntSequenceGenerator",
      "com.fasterxml.jackson.databind.AbstractTypeResolver",
      "com.fasterxml.jackson.databind.module.SimpleAbstractTypeResolver",
      "com.fasterxml.jackson.databind.deser.UnresolvedId",
      "com.fasterxml.jackson.databind.type.ClassKey",
      "com.fasterxml.jackson.databind.util.ObjectBuffer",
      "com.fasterxml.jackson.databind.ObjectMapper$2",
      "com.fasterxml.jackson.databind.introspect.VirtualAnnotatedMember",
      "com.fasterxml.jackson.databind.JsonSerializable$Base",
      "com.fasterxml.jackson.databind.JsonNode",
      "com.fasterxml.jackson.databind.node.BaseJsonNode",
      "com.fasterxml.jackson.databind.node.ValueNode",
      "com.fasterxml.jackson.databind.node.BooleanNode",
      "com.fasterxml.jackson.core.base.ParserMinimalBase",
      "com.fasterxml.jackson.databind.node.TreeTraversingParser",
      "com.fasterxml.jackson.databind.node.NodeCursor",
      "com.fasterxml.jackson.databind.node.NodeCursor$RootCursor",
      "com.fasterxml.jackson.core.util.BufferRecyclers",
      "com.fasterxml.jackson.core.json.ByteSourceJsonBootstrapper",
      "com.fasterxml.jackson.core.base.ParserBase",
      "com.fasterxml.jackson.core.json.UTF8StreamJsonParser",
      "com.fasterxml.jackson.core.util.TextBuffer",
      "com.fasterxml.jackson.core.json.JsonReadContext",
      "com.fasterxml.jackson.databind.type.ResolvedRecursiveType",
      "com.fasterxml.jackson.databind.type.TypeBindings$TypeParamStash",
      "com.fasterxml.jackson.databind.type.TypeBindings$AsKey",
      "com.fasterxml.jackson.databind.exc.MismatchedInputException",
      "com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper$Base",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$9",
      "com.fasterxml.jackson.annotation.JsonProperty$Access",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$10",
      "com.fasterxml.jackson.databind.ext.OptionalHandlerFactory",
      "com.fasterxml.jackson.databind.ser.PropertyBuilder",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$3",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$2",
      "com.fasterxml.jackson.databind.BeanProperty$Std",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$4",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$5",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$6",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$7",
      "com.fasterxml.jackson.databind.PropertyMetadata",
      "com.fasterxml.jackson.databind.ser.PropertyBuilder$1",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$1",
      "com.fasterxml.jackson.databind.ser.PropertyWriter",
      "com.fasterxml.jackson.databind.ser.BeanPropertyWriter",
      "com.fasterxml.jackson.databind.ser.impl.PropertySerializerMap",
      "com.fasterxml.jackson.databind.ser.impl.PropertySerializerMap$Empty",
      "com.fasterxml.jackson.annotation.JsonIgnoreProperties$Value",
      "com.fasterxml.jackson.databind.ser.std.BeanSerializerBase",
      "com.fasterxml.jackson.databind.ser.BeanSerializer",
      "com.fasterxml.jackson.databind.introspect.AnnotationCollector$OneCollector",
      "com.fasterxml.jackson.databind.type.CollectionType",
      "com.fasterxml.jackson.databind.ser.std.ClassSerializer",
      "com.fasterxml.jackson.core.json.WriterBasedJsonGenerator",
      "com.fasterxml.jackson.core.json.DupDetector",
      "com.fasterxml.jackson.databind.node.ContainerNode",
      "com.fasterxml.jackson.databind.node.ArrayNode",
      "com.fasterxml.jackson.databind.node.NumericNode",
      "com.fasterxml.jackson.databind.node.IntNode",
      "com.fasterxml.jackson.databind.ser.impl.ReadOnlyClassToSerializerMap$Bucket",
      "com.fasterxml.jackson.databind.ser.std.SerializableSerializer",
      "com.fasterxml.jackson.core.io.NumberOutput",
      "com.fasterxml.jackson.databind.node.MissingNode",
      "com.fasterxml.jackson.databind.module.SimpleDeserializers"
    );
  }
}
