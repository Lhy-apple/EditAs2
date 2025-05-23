/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Wed Jul 12 13:36:22 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class StdTypeResolverBuilder_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "com.fasterxml.jackson.databind.jsontype.impl.StdTypeResolverBuilder"; 
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
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(StdTypeResolverBuilder_ESTest_scaffolding.class.getClassLoader() ,
      "com.fasterxml.jackson.core.util.JsonParserSequence",
      "com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver",
      "com.fasterxml.jackson.databind.ObjectMapper$DefaultTypeResolverBuilder",
      "com.fasterxml.jackson.databind.jsontype.TypeDeserializer",
      "com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeSerializer",
      "com.fasterxml.jackson.core.Versioned",
      "com.fasterxml.jackson.databind.SerializationConfig",
      "com.fasterxml.jackson.databind.jsontype.TypeSerializer",
      "com.fasterxml.jackson.databind.deser.std.StdDeserializer",
      "com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer",
      "com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeSerializer",
      "com.fasterxml.jackson.databind.type.TypeFactory",
      "com.fasterxml.jackson.databind.cfg.MapperConfigBase",
      "com.fasterxml.jackson.databind.JsonSerializable",
      "com.fasterxml.jackson.databind.type.ArrayType",
      "com.fasterxml.jackson.databind.util.LRUMap",
      "com.fasterxml.jackson.databind.jsontype.NamedType",
      "com.fasterxml.jackson.databind.jsontype.impl.TypeNameIdResolver",
      "com.fasterxml.jackson.databind.jsontype.TypeResolverBuilder",
      "com.fasterxml.jackson.databind.jsontype.impl.StdTypeResolverBuilder$1",
      "com.fasterxml.jackson.core.type.TypeReference",
      "com.fasterxml.jackson.databind.JsonDeserializer",
      "com.fasterxml.jackson.databind.type.MapLikeType",
      "com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeSerializer",
      "com.fasterxml.jackson.databind.type.MapType",
      "com.fasterxml.jackson.databind.deser.std.NullifyingDeserializer",
      "com.fasterxml.jackson.databind.util.Named",
      "com.fasterxml.jackson.core.util.JsonParserDelegate",
      "com.fasterxml.jackson.databind.ObjectMapper$2",
      "com.fasterxml.jackson.databind.BeanProperty",
      "com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeSerializer",
      "com.fasterxml.jackson.databind.type.TypeBase",
      "com.fasterxml.jackson.databind.type.CollectionType",
      "com.fasterxml.jackson.annotation.JsonTypeInfo$Id",
      "com.fasterxml.jackson.databind.type.TypeModifier",
      "com.fasterxml.jackson.annotation.JsonTypeInfo",
      "com.fasterxml.jackson.databind.jsontype.impl.StdTypeResolverBuilder",
      "com.fasterxml.jackson.databind.cfg.MapperConfig",
      "com.fasterxml.jackson.databind.jsontype.impl.AsExistingPropertyTypeSerializer",
      "com.fasterxml.jackson.databind.introspect.ClassIntrospector$MixInResolver",
      "com.fasterxml.jackson.databind.jsontype.impl.TypeSerializerBase",
      "com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase",
      "com.fasterxml.jackson.databind.type.TypeParser",
      "com.fasterxml.jackson.core.JsonGenerator",
      "com.fasterxml.jackson.annotation.JsonTypeInfo$As",
      "com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer",
      "com.fasterxml.jackson.core.ObjectCodec",
      "com.fasterxml.jackson.databind.type.SimpleType",
      "com.fasterxml.jackson.databind.type.ClassStack",
      "com.fasterxml.jackson.databind.DatabindContext",
      "com.fasterxml.jackson.core.type.ResolvedType",
      "com.fasterxml.jackson.databind.DeserializationConfig",
      "com.fasterxml.jackson.databind.type.TypeBindings",
      "com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer",
      "com.fasterxml.jackson.databind.JavaType",
      "com.fasterxml.jackson.databind.jsontype.TypeIdResolver",
      "com.fasterxml.jackson.databind.type.CollectionLikeType",
      "com.fasterxml.jackson.core.TreeCodec",
      "com.fasterxml.jackson.databind.jsontype.impl.TypeIdResolverBase",
      "com.fasterxml.jackson.databind.DeserializationContext",
      "com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer",
      "com.fasterxml.jackson.core.JsonParser",
      "com.fasterxml.jackson.databind.ObjectMapper",
      "com.fasterxml.jackson.databind.type.ResolvedRecursiveType",
      "com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver",
      "com.fasterxml.jackson.annotation.JsonTypeInfo$None",
      "com.fasterxml.jackson.databind.type.ReferenceType",
      "com.fasterxml.jackson.databind.ObjectMapper$DefaultTyping",
      "com.fasterxml.jackson.databind.annotation.NoClass"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(StdTypeResolverBuilder_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "com.fasterxml.jackson.databind.jsontype.impl.StdTypeResolverBuilder",
      "com.fasterxml.jackson.annotation.JsonTypeInfo$Id",
      "com.fasterxml.jackson.annotation.JsonTypeInfo$As",
      "com.fasterxml.jackson.databind.jsontype.impl.StdTypeResolverBuilder$1",
      "com.fasterxml.jackson.annotation.JsonAutoDetect$Visibility",
      "com.fasterxml.jackson.databind.ObjectMapper$DefaultTypeResolverBuilder",
      "com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector",
      "com.fasterxml.jackson.databind.BeanDescription",
      "com.fasterxml.jackson.databind.introspect.BasicBeanDescription",
      "com.fasterxml.jackson.databind.ser.BeanSerializerBuilder",
      "com.fasterxml.jackson.databind.ObjectMapper$2",
      "com.fasterxml.jackson.databind.jsontype.SubtypeResolver",
      "com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver",
      "com.fasterxml.jackson.databind.util.LRUMap",
      "com.fasterxml.jackson.databind.type.TypeParser",
      "com.fasterxml.jackson.databind.type.TypeBindings",
      "com.fasterxml.jackson.core.type.ResolvedType",
      "com.fasterxml.jackson.databind.JavaType",
      "com.fasterxml.jackson.databind.type.TypeBase",
      "com.fasterxml.jackson.databind.type.SimpleType",
      "com.fasterxml.jackson.databind.type.TypeFactory",
      "com.fasterxml.jackson.databind.introspect.TypeResolutionContext$Basic",
      "com.fasterxml.jackson.databind.type.CollectionLikeType",
      "com.fasterxml.jackson.databind.type.CollectionType",
      "com.fasterxml.jackson.databind.introspect.SimpleMixInResolver",
      "com.fasterxml.jackson.databind.util.RootNameLookup",
      "com.fasterxml.jackson.databind.cfg.ConfigOverrides",
      "com.fasterxml.jackson.annotation.JsonInclude$Include",
      "com.fasterxml.jackson.annotation.JsonInclude$Value",
      "com.fasterxml.jackson.annotation.JsonFormat$Shape",
      "com.fasterxml.jackson.annotation.JsonFormat$Features",
      "com.fasterxml.jackson.annotation.JsonFormat$Value",
      "com.fasterxml.jackson.databind.cfg.MapperConfig",
      "com.fasterxml.jackson.databind.cfg.MapperConfigBase",
      "com.fasterxml.jackson.core.io.SerializedString",
      "com.fasterxml.jackson.core.util.DefaultPrettyPrinter",
      "com.fasterxml.jackson.core.util.DefaultPrettyPrinter$NopIndenter",
      "com.fasterxml.jackson.core.util.DefaultPrettyPrinter$FixedSpaceIndenter",
      "com.fasterxml.jackson.core.util.DefaultIndenter",
      "com.fasterxml.jackson.databind.SerializationConfig",
      "com.fasterxml.jackson.databind.cfg.ContextAttributes",
      "com.fasterxml.jackson.databind.cfg.ContextAttributes$Impl",
      "com.fasterxml.jackson.databind.DatabindContext",
      "com.fasterxml.jackson.databind.JsonSerializer",
      "com.fasterxml.jackson.databind.ser.std.StdSerializer",
      "com.fasterxml.jackson.databind.ser.impl.FailingSerializer",
      "com.fasterxml.jackson.databind.ser.impl.UnknownSerializer",
      "com.fasterxml.jackson.databind.SerializerProvider",
      "com.fasterxml.jackson.databind.ser.DefaultSerializerProvider",
      "com.fasterxml.jackson.databind.ser.DefaultSerializerProvider$Impl",
      "com.fasterxml.jackson.databind.ser.std.NullSerializer",
      "com.fasterxml.jackson.databind.ser.SerializerCache",
      "com.fasterxml.jackson.databind.type.ReferenceType",
      "com.fasterxml.jackson.databind.deser.DeserializerFactory",
      "com.fasterxml.jackson.databind.PropertyName",
      "com.fasterxml.jackson.databind.deser.BasicDeserializerFactory",
      "com.fasterxml.jackson.databind.deser.std.StdKeyDeserializers",
      "com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig",
      "com.fasterxml.jackson.databind.deser.BeanDeserializerFactory",
      "com.fasterxml.jackson.databind.DeserializationContext",
      "com.fasterxml.jackson.databind.deser.DefaultDeserializationContext",
      "com.fasterxml.jackson.databind.deser.DefaultDeserializationContext$Impl",
      "com.fasterxml.jackson.databind.deser.DeserializerCache",
      "com.fasterxml.jackson.databind.type.TypeModifier",
      "com.fasterxml.jackson.databind.type.MapLikeType",
      "com.fasterxml.jackson.databind.type.ClassStack",
      "com.fasterxml.jackson.databind.util.ClassUtil$EmptyIterator",
      "com.fasterxml.jackson.databind.util.ClassUtil",
      "com.fasterxml.jackson.databind.type.TypeBindings$TypeParamStash",
      "com.fasterxml.jackson.databind.type.TypeBindings$AsKey",
      "com.fasterxml.jackson.databind.jsontype.impl.TypeIdResolverBase",
      "com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver",
      "com.fasterxml.jackson.databind.type.ResolvedRecursiveType",
      "com.fasterxml.jackson.databind.type.TypeParser$MyTokenizer",
      "com.fasterxml.jackson.databind.DeserializationConfig",
      "com.fasterxml.jackson.databind.node.JsonNodeFactory",
      "com.fasterxml.jackson.databind.cfg.HandlerInstantiator",
      "com.fasterxml.jackson.databind.type.ArrayType",
      "com.fasterxml.jackson.core.JsonFactory$Feature",
      "com.fasterxml.jackson.core.JsonFactory",
      "com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer",
      "com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer",
      "com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer$TableInfo",
      "com.fasterxml.jackson.databind.jsontype.NamedType",
      "com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver",
      "com.fasterxml.jackson.databind.type.MapType",
      "com.fasterxml.jackson.databind.module.SimpleKeyDeserializers",
      "com.fasterxml.jackson.databind.util.ArrayBuilders",
      "com.fasterxml.jackson.databind.jsontype.impl.TypeNameIdResolver",
      "com.fasterxml.jackson.databind.JsonDeserializer",
      "com.fasterxml.jackson.databind.deser.std.StdDeserializer",
      "com.fasterxml.jackson.databind.deser.std.StdScalarDeserializer",
      "com.fasterxml.jackson.databind.deser.std.FromStringDeserializer",
      "com.fasterxml.jackson.databind.ext.CoreXMLDeserializers$Std",
      "com.fasterxml.jackson.core.TreeCodec",
      "com.fasterxml.jackson.core.ObjectCodec",
      "com.fasterxml.jackson.databind.AnnotationIntrospector",
      "com.fasterxml.jackson.databind.ext.Java7SupportImpl",
      "com.fasterxml.jackson.databind.ext.Java7Support",
      "com.fasterxml.jackson.databind.introspect.JacksonAnnotationIntrospector",
      "com.fasterxml.jackson.databind.introspect.VisibilityChecker$Std",
      "com.fasterxml.jackson.databind.cfg.BaseSettings",
      "com.fasterxml.jackson.databind.util.StdDateFormat",
      "com.fasterxml.jackson.core.Base64Variant",
      "com.fasterxml.jackson.core.Base64Variants",
      "com.fasterxml.jackson.databind.ObjectMapper",
      "com.fasterxml.jackson.databind.MappingJsonFactory",
      "com.fasterxml.jackson.databind.introspect.ClassIntrospector",
      "com.fasterxml.jackson.databind.introspect.Annotated",
      "com.fasterxml.jackson.databind.introspect.AnnotatedClass",
      "com.fasterxml.jackson.databind.introspect.BasicClassIntrospector",
      "com.fasterxml.jackson.databind.ser.SerializerFactory",
      "com.fasterxml.jackson.databind.ser.std.StdScalarSerializer",
      "com.fasterxml.jackson.databind.ser.std.NonTypedScalarSerializerBase",
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
      "com.fasterxml.jackson.databind.ObjectReader",
      "com.fasterxml.jackson.databind.deser.ValueInstantiators$Base",
      "com.fasterxml.jackson.databind.module.SimpleValueInstantiators",
      "com.fasterxml.jackson.databind.introspect.NopAnnotationIntrospector$1",
      "com.fasterxml.jackson.databind.introspect.NopAnnotationIntrospector",
      "com.fasterxml.jackson.databind.introspect.BeanPropertyDefinition",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder",
      "com.fasterxml.jackson.databind.introspect.AnnotationMap",
      "com.fasterxml.jackson.databind.jsontype.TypeDeserializer",
      "com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase",
      "com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer",
      "com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer",
      "com.fasterxml.jackson.annotation.SimpleObjectIdResolver",
      "com.fasterxml.jackson.databind.AbstractTypeResolver",
      "com.fasterxml.jackson.databind.module.SimpleAbstractTypeResolver",
      "com.fasterxml.jackson.databind.type.ClassKey",
      "com.fasterxml.jackson.databind.InjectableValues",
      "com.fasterxml.jackson.databind.InjectableValues$Std",
      "com.fasterxml.jackson.core.type.TypeReference",
      "com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer",
      "com.fasterxml.jackson.databind.deser.ValueInstantiator",
      "com.fasterxml.jackson.databind.deser.ValueInstantiator$Base",
      "com.fasterxml.jackson.databind.PropertyNamingStrategy$PropertyNamingStrategyBase",
      "com.fasterxml.jackson.databind.PropertyNamingStrategy$SnakeCaseStrategy",
      "com.fasterxml.jackson.databind.PropertyNamingStrategy$UpperCamelCaseStrategy",
      "com.fasterxml.jackson.databind.PropertyNamingStrategy$LowerCaseStrategy",
      "com.fasterxml.jackson.databind.PropertyNamingStrategy$KebabCaseStrategy",
      "com.fasterxml.jackson.databind.PropertyNamingStrategy",
      "com.fasterxml.jackson.databind.deser.BeanDeserializerBuilder",
      "com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer",
      "com.fasterxml.jackson.databind.ser.FilterProvider",
      "com.fasterxml.jackson.databind.ser.impl.SimpleFilterProvider",
      "com.fasterxml.jackson.databind.deser.Deserializers$Base",
      "com.fasterxml.jackson.databind.introspect.AnnotationIntrospectorPair",
      "com.fasterxml.jackson.databind.jsontype.TypeSerializer",
      "com.fasterxml.jackson.databind.jsontype.impl.TypeSerializerBase",
      "com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeSerializer",
      "com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeSerializer",
      "com.fasterxml.jackson.databind.jsontype.impl.AsExistingPropertyTypeSerializer",
      "com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeSerializer",
      "com.fasterxml.jackson.annotation.ObjectIdGenerator",
      "com.fasterxml.jackson.annotation.ObjectIdGenerators$Base",
      "com.fasterxml.jackson.annotation.ObjectIdGenerators$StringIdGenerator",
      "com.fasterxml.jackson.databind.deser.BeanDeserializerModifier",
      "com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeSerializer",
      "com.fasterxml.jackson.core.io.IOContext",
      "com.fasterxml.jackson.core.util.BufferRecycler",
      "com.fasterxml.jackson.core.json.ByteSourceJsonBootstrapper",
      "com.fasterxml.jackson.core.JsonParser",
      "com.fasterxml.jackson.core.base.ParserMinimalBase",
      "com.fasterxml.jackson.core.base.ParserBase",
      "com.fasterxml.jackson.core.io.CharTypes",
      "com.fasterxml.jackson.core.json.UTF8StreamJsonParser",
      "com.fasterxml.jackson.core.util.TextBuffer",
      "com.fasterxml.jackson.core.JsonStreamContext",
      "com.fasterxml.jackson.core.json.JsonReadContext",
      "com.fasterxml.jackson.core.JsonGenerator",
      "com.fasterxml.jackson.databind.util.TokenBuffer",
      "com.fasterxml.jackson.core.json.JsonWriteContext",
      "com.fasterxml.jackson.core.JsonToken",
      "com.fasterxml.jackson.databind.util.TokenBuffer$Segment",
      "com.fasterxml.jackson.core.util.InternCache",
      "com.fasterxml.jackson.databind.introspect.VisibilityChecker$1",
      "com.fasterxml.jackson.databind.deser.impl.ObjectIdReader",
      "com.fasterxml.jackson.annotation.ObjectIdGenerators$UUIDGenerator",
      "com.fasterxml.jackson.databind.introspect.AnnotatedMember",
      "com.fasterxml.jackson.databind.introspect.VirtualAnnotatedMember",
      "com.fasterxml.jackson.databind.introspect.AnnotatedField",
      "com.fasterxml.jackson.annotation.ObjectIdGenerators$IntSequenceGenerator",
      "com.fasterxml.jackson.databind.util.ArrayIterator",
      "com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer",
      "com.fasterxml.jackson.databind.deser.impl.CreatorCollector",
      "com.fasterxml.jackson.databind.util.ClassUtil$Ctor",
      "com.fasterxml.jackson.databind.introspect.AnnotatedWithParams",
      "com.fasterxml.jackson.databind.introspect.AnnotatedConstructor",
      "com.fasterxml.jackson.databind.introspect.AnnotatedMethod",
      "com.fasterxml.jackson.annotation.JsonCreator$Mode",
      "com.fasterxml.jackson.databind.introspect.AnnotatedParameter",
      "com.fasterxml.jackson.annotation.JsonAutoDetect$1",
      "com.fasterxml.jackson.databind.deser.impl.CreatorCollector$StdTypeConstructor",
      "com.fasterxml.jackson.databind.deser.std.StdValueInstantiator",
      "com.fasterxml.jackson.databind.deser.std.ContainerDeserializerBase",
      "com.fasterxml.jackson.databind.deser.std.CollectionDeserializer",
      "com.fasterxml.jackson.databind.deser.std.MapDeserializer",
      "com.fasterxml.jackson.annotation.JsonIgnoreProperties$Value",
      "com.fasterxml.jackson.databind.deser.std.StringDeserializer",
      "com.fasterxml.jackson.databind.deser.std.NumberDeserializers",
      "com.fasterxml.jackson.databind.deser.std.NumberDeserializers$NumberDeserializer",
      "com.fasterxml.jackson.databind.util.LinkedNode",
      "com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer$Vanilla",
      "com.fasterxml.jackson.core.json.ReaderBasedJsonParser",
      "com.fasterxml.jackson.databind.deser.DeserializationProblemHandler",
      "com.fasterxml.jackson.core.util.VersionUtil",
      "com.fasterxml.jackson.core.Version",
      "com.fasterxml.jackson.databind.cfg.PackageVersion",
      "com.fasterxml.jackson.databind.ser.impl.ReadOnlyClassToSerializerMap",
      "com.fasterxml.jackson.databind.deser.AbstractDeserializer",
      "com.fasterxml.jackson.databind.ser.impl.ObjectIdWriter",
      "com.fasterxml.jackson.databind.JsonSerializable$Base",
      "com.fasterxml.jackson.databind.JsonNode",
      "com.fasterxml.jackson.databind.node.BaseJsonNode",
      "com.fasterxml.jackson.databind.node.ValueNode",
      "com.fasterxml.jackson.databind.node.BinaryNode",
      "com.fasterxml.jackson.core.filter.TokenFilter",
      "com.fasterxml.jackson.core.util.JsonGeneratorDelegate",
      "com.fasterxml.jackson.core.filter.FilteringGeneratorDelegate",
      "com.fasterxml.jackson.core.filter.TokenFilterContext",
      "com.fasterxml.jackson.databind.deser.DataFormatReaders",
      "com.fasterxml.jackson.core.format.MatchStrength",
      "com.fasterxml.jackson.core.JsonProcessingException",
      "com.fasterxml.jackson.databind.JsonMappingException",
      "com.fasterxml.jackson.core.JsonLocation",
      "com.fasterxml.jackson.databind.introspect.AnnotatedMethodMap",
      "com.fasterxml.jackson.databind.module.SimpleDeserializers",
      "com.fasterxml.jackson.databind.deser.BeanDeserializerBase",
      "com.fasterxml.jackson.databind.deser.BeanDeserializer",
      "com.fasterxml.jackson.core.util.ByteArrayBuilder",
      "com.fasterxml.jackson.core.base.GeneratorBase",
      "com.fasterxml.jackson.core.json.JsonGeneratorImpl",
      "com.fasterxml.jackson.core.json.UTF8JsonGenerator",
      "com.fasterxml.jackson.databind.ext.OptionalHandlerFactory",
      "com.fasterxml.jackson.databind.deser.std.JdkDeserializers",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$Linked",
      "com.fasterxml.jackson.databind.introspect.MemberKey",
      "com.fasterxml.jackson.databind.util.BeanUtil",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$9",
      "com.fasterxml.jackson.annotation.JsonProperty$Access",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$10",
      "com.fasterxml.jackson.databind.deser.impl.BeanPropertyMap",
      "com.fasterxml.jackson.annotation.JsonFormat$Feature",
      "com.fasterxml.jackson.databind.BeanProperty",
      "com.fasterxml.jackson.core.json.DupDetector",
      "com.fasterxml.jackson.databind.exc.InvalidFormatException",
      "com.fasterxml.jackson.core.io.OutputDecorator",
      "com.fasterxml.jackson.databind.deser.std.FromStringDeserializer$Std",
      "com.fasterxml.jackson.core.io.InputDecorator",
      "com.fasterxml.jackson.core.json.WriterBasedJsonGenerator",
      "com.fasterxml.jackson.databind.exc.InvalidTypeIdException",
      "com.fasterxml.jackson.databind.introspect.ConcreteBeanPropertyBase",
      "com.fasterxml.jackson.databind.deser.impl.FailingDeserializer",
      "com.fasterxml.jackson.databind.deser.SettableBeanProperty",
      "com.fasterxml.jackson.databind.deser.impl.SetterlessProperty",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$4",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$5",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$6",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$7",
      "com.fasterxml.jackson.databind.PropertyMetadata",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$1",
      "com.fasterxml.jackson.databind.deser.std.StringCollectionDeserializer",
      "com.fasterxml.jackson.core.io.CharacterEscapes",
      "com.fasterxml.jackson.core.JsonpCharacterEscapes",
      "com.fasterxml.jackson.databind.introspect.ObjectIdInfo",
      "com.fasterxml.jackson.databind.util.ObjectBuffer",
      "com.fasterxml.jackson.databind.Module",
      "com.fasterxml.jackson.databind.module.SimpleModule",
      "com.fasterxml.jackson.databind.ObjectMapper$1",
      "com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$8",
      "com.fasterxml.jackson.core.JsonPointer",
      "com.fasterxml.jackson.annotation.ObjectIdGenerator$IdKey"
    );
  }
}
