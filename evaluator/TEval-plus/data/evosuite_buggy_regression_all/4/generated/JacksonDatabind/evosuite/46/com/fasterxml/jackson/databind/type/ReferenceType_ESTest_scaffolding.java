/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Tue Sep 26 21:41:13 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class ReferenceType_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "com.fasterxml.jackson.databind.type.ReferenceType"; 
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
    java.lang.System.setProperty("user.dir", "/data/lhy/TEval-plus"); 
    java.lang.System.setProperty("java.io.tmpdir", "/tmp"); 
  }

  private static void initializeClasses() {
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(ReferenceType_ESTest_scaffolding.class.getClassLoader() ,
      "com.fasterxml.jackson.core.JsonGenerator",
      "com.fasterxml.jackson.databind.type.TypeParser",
      "com.fasterxml.jackson.core.Versioned",
      "com.fasterxml.jackson.databind.type.SimpleType",
      "com.fasterxml.jackson.databind.DatabindContext",
      "com.fasterxml.jackson.core.type.ResolvedType",
      "com.fasterxml.jackson.databind.jsontype.TypeSerializer",
      "com.fasterxml.jackson.databind.type.ClassKey",
      "com.fasterxml.jackson.databind.type.TypeFactory",
      "com.fasterxml.jackson.databind.type.TypeBindings",
      "com.fasterxml.jackson.databind.JsonSerializable",
      "com.fasterxml.jackson.databind.type.ArrayType",
      "com.fasterxml.jackson.databind.type.HierarchicType",
      "com.fasterxml.jackson.databind.util.LRUMap",
      "com.fasterxml.jackson.databind.JavaType",
      "com.fasterxml.jackson.databind.SerializerProvider",
      "com.fasterxml.jackson.core.type.TypeReference",
      "com.fasterxml.jackson.databind.type.MapLikeType",
      "com.fasterxml.jackson.databind.type.MapType",
      "com.fasterxml.jackson.databind.type.CollectionLikeType",
      "com.fasterxml.jackson.databind.type.TypeBase",
      "com.fasterxml.jackson.databind.type.CollectionType",
      "com.fasterxml.jackson.core.JsonProcessingException",
      "com.fasterxml.jackson.databind.type.TypeModifier",
      "com.fasterxml.jackson.databind.type.ReferenceType"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(ReferenceType_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "com.fasterxml.jackson.core.type.ResolvedType",
      "com.fasterxml.jackson.databind.JavaType",
      "com.fasterxml.jackson.databind.type.TypeBase",
      "com.fasterxml.jackson.databind.type.SimpleType",
      "com.fasterxml.jackson.databind.type.ReferenceType",
      "com.fasterxml.jackson.annotation.JsonAutoDetect$Visibility",
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
      "com.fasterxml.jackson.databind.type.MapLikeType",
      "com.fasterxml.jackson.databind.util.LRUMap",
      "com.fasterxml.jackson.databind.type.TypeParser",
      "com.fasterxml.jackson.databind.type.TypeFactory",
      "com.fasterxml.jackson.databind.type.MapType",
      "com.fasterxml.jackson.databind.type.ArrayType",
      "com.fasterxml.jackson.databind.type.TypeBindings",
      "com.fasterxml.jackson.databind.type.CollectionLikeType",
      "com.fasterxml.jackson.databind.type.CollectionType",
      "com.fasterxml.jackson.databind.deser.std.StdKeyDeserializers",
      "com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig",
      "com.fasterxml.jackson.databind.deser.DeserializerFactory",
      "com.fasterxml.jackson.databind.PropertyName",
      "com.fasterxml.jackson.databind.deser.BasicDeserializerFactory",
      "com.fasterxml.jackson.databind.deser.BeanDeserializerFactory",
      "com.fasterxml.jackson.databind.DeserializationContext",
      "com.fasterxml.jackson.databind.deser.DefaultDeserializationContext",
      "com.fasterxml.jackson.databind.deser.DefaultDeserializationContext$Impl",
      "com.fasterxml.jackson.databind.deser.DeserializerCache",
      "com.fasterxml.jackson.databind.type.TypeModifier",
      "com.fasterxml.jackson.databind.type.TypeParser$MyTokenizer",
      "com.fasterxml.jackson.databind.type.ClassKey",
      "com.fasterxml.jackson.databind.util.ArrayBuilders",
      "com.fasterxml.jackson.databind.util.ClassUtil",
      "com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector",
      "com.fasterxml.jackson.databind.BeanDescription",
      "com.fasterxml.jackson.databind.introspect.BasicBeanDescription",
      "com.fasterxml.jackson.databind.type.HierarchicType",
      "com.fasterxml.jackson.databind.deser.BeanDeserializerBuilder"
    );
  }
}