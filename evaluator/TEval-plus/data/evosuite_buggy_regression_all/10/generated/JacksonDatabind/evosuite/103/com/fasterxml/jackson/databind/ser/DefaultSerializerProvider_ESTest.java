/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:45:26 GMT 2023
 */

package com.fasterxml.jackson.databind.ser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializationConfig;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.cfg.ConfigOverrides;
import com.fasterxml.jackson.databind.cfg.SerializerFactoryConfig;
import com.fasterxml.jackson.databind.ext.DOMSerializer;
import com.fasterxml.jackson.databind.ext.NioPathSerializer;
import com.fasterxml.jackson.databind.introspect.Annotated;
import com.fasterxml.jackson.databind.introspect.BeanPropertyDefinition;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.ser.BeanSerializer;
import com.fasterxml.jackson.databind.ser.BeanSerializerFactory;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.impl.WritableObjectId;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.RootNameLookup;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DefaultSerializerProvider_ESTest extends DefaultSerializerProvider_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      ConfigOverrides configOverrides0 = new ConfigOverrides();
      SerializationConfig serializationConfig0 = new SerializationConfig((BaseSettings) null, stdSubtypeResolver0, (SimpleMixInResolver) null, rootNameLookup0, configOverrides0);
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig0);
      DefaultSerializerProvider defaultSerializerProvider0 = defaultSerializerProvider_Impl0.createInstance(serializationConfig0, beanSerializerFactory0);
      ObjectIdGenerator<NioPathSerializer> objectIdGenerator0 = (ObjectIdGenerator<NioPathSerializer>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      doReturn((ObjectIdGenerator) null).when(objectIdGenerator0).newForSerialization(any());
      WritableObjectId writableObjectId0 = defaultSerializerProvider0.findObjectId(defaultSerializerProvider_Impl0, objectIdGenerator0);
      assertNotNull(writableObjectId0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      DefaultSerializerProvider defaultSerializerProvider0 = defaultSerializerProvider_Impl0.copy();
      assertNotSame(defaultSerializerProvider_Impl0, defaultSerializerProvider0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      try { 
        defaultSerializerProvider_Impl0.serializeValue((JsonGenerator) null, (Object) null, javaType0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // [no message for java.lang.NullPointerException]
         //
         verifyException("com.fasterxml.jackson.databind.ser.DefaultSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      defaultSerializerProvider_Impl0.flushCachedSerializers();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      int int0 = defaultSerializerProvider_Impl0.cachedSerializersCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Map> class0 = Map.class;
      Class<ObjectIdGenerators.StringIdGenerator> class1 = ObjectIdGenerators.StringIdGenerator.class;
      Class<BeanSerializer> class2 = BeanSerializer.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class1, class2);
      MockFile mockFile0 = new MockFile("8.LHi_hx", "Cannot coerce %s to Null value %s (%s `%s.%s` to allow)");
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_BE;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((File) mockFile0, jsonEncoding0);
      try { 
        defaultSerializerProvider_Impl0.serializeValue(jsonGenerator0, (Object) class2, (JavaType) mapType0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Incompatible types: declared root type ([map type; class java.util.Map, [simple type, class com.fasterxml.jackson.annotation.ObjectIdGenerators$StringIdGenerator] -> [simple type, class com.fasterxml.jackson.databind.ser.BeanSerializer]]) vs `java.lang.Class`
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidDefinitionException", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      DOMSerializer dOMSerializer0 = new DOMSerializer();
      JsonSerializer<Object> jsonSerializer0 = defaultSerializerProvider_Impl0.serializerInstance((Annotated) null, dOMSerializer0);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonSerializer<Object> jsonSerializer0 = defaultSerializerProvider_Impl0.serializerInstance((Annotated) null, (Object) null);
      assertNull(jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<BeanSerializer> class0 = BeanSerializer.class;
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.serializerInstance((Annotated) null, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.DefaultSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JavaType javaType0 = TypeFactory.unknownType();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.serializerInstance((Annotated) null, javaType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.DefaultSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<String> class0 = String.class;
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.includeFilterInstance((BeanPropertyDefinition) null, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.DefaultSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      boolean boolean0 = defaultSerializerProvider_Impl0.includeFilterSuppressNulls(defaultSerializerProvider_Impl0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      boolean boolean0 = defaultSerializerProvider_Impl0.includeFilterSuppressNulls((Object) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectIdGenerator<NioPathSerializer> objectIdGenerator0 = (ObjectIdGenerator<NioPathSerializer>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.findObjectId(defaultSerializerProvider_Impl0, objectIdGenerator0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<String> class0 = String.class;
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.hasSerializerFor(class0, (AtomicReference<Throwable>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<Object> class0 = Object.class;
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.hasSerializerFor(class0, atomicReference0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.DefaultSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<ObjectIdGenerators.StringIdGenerator> class0 = ObjectIdGenerators.StringIdGenerator.class;
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      boolean boolean0 = defaultSerializerProvider_Impl0.hasSerializerFor(class0, atomicReference0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectIdGenerators.StringIdGenerator objectIdGenerators_StringIdGenerator0 = new ObjectIdGenerators.StringIdGenerator();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.serializeValue((JsonGenerator) null, (Object) objectIdGenerators_StringIdGenerator0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      try { 
        defaultSerializerProvider_Impl0.serializeValue((JsonGenerator) null, (Object) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // [no message for java.lang.NullPointerException]
         //
         verifyException("com.fasterxml.jackson.databind.ser.DefaultSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      ObjectIdGenerators.StringIdGenerator objectIdGenerators_StringIdGenerator0 = new ObjectIdGenerators.StringIdGenerator();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.serializeValue((JsonGenerator) null, (Object) objectIdGenerators_StringIdGenerator0, javaType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<BeanSerializer> class0 = BeanSerializer.class;
      Class<ObjectNode> class1 = ObjectNode.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class1);
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base(defaultSerializerProvider_Impl0);
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.acceptJsonFormatVisitor(mapLikeType0, jsonFormatVisitorWrapper_Base0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base(defaultSerializerProvider_Impl0);
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.acceptJsonFormatVisitor((JavaType) null, jsonFormatVisitorWrapper_Base0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // A class must be provided
         //
         verifyException("com.fasterxml.jackson.databind.ser.DefaultSerializerProvider", e);
      }
  }
}
