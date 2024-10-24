/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:14:56 GMT 2023
 */

package com.fasterxml.jackson.databind.ser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializationConfig;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.cfg.ConfigOverrides;
import com.fasterxml.jackson.databind.cfg.SerializerFactoryConfig;
import com.fasterxml.jackson.databind.ext.CoreXMLSerializers;
import com.fasterxml.jackson.databind.introspect.Annotated;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import com.fasterxml.jackson.databind.ser.BeanSerializer;
import com.fasterxml.jackson.databind.ser.BeanSerializerFactory;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.util.RootNameLookup;
import java.io.IOException;
import java.sql.ClientInfoStatus;
import java.sql.SQLClientInfoException;
import java.sql.SQLRecoverableException;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicReference;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DefaultSerializerProvider_ESTest extends DefaultSerializerProvider_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanSerializerFactory beanSerializerFactory0 = BeanSerializerFactory.instance;
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.createInstance((SerializationConfig) null, beanSerializerFactory0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      ConfigOverrides configOverrides0 = new ConfigOverrides();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, (SimpleMixInResolver) null, rootNameLookup0, configOverrides0);
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver(deserializationConfig0);
      SerializationConfig serializationConfig0 = new SerializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0, configOverrides0);
      SerializerFactoryConfig serializerFactoryConfig0 = new SerializerFactoryConfig();
      BeanSerializerFactory beanSerializerFactory0 = new BeanSerializerFactory(serializerFactoryConfig0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl1 = new DefaultSerializerProvider.Impl(defaultSerializerProvider_Impl0, serializationConfig0, beanSerializerFactory0);
      assertTrue(defaultSerializerProvider_Impl1.canOverrideAccessModifiers());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
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
      JsonGenerator jsonGenerator0 = defaultSerializerProvider_Impl0.getGenerator();
      assertNull(jsonGenerator0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<CoreXMLSerializers.XMLGregorianCalendarSerializer> class0 = CoreXMLSerializers.XMLGregorianCalendarSerializer.class;
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
  public void test07()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonSerializer<Object> jsonSerializer0 = defaultSerializerProvider_Impl0.serializerInstance((Annotated) null, (Object) null);
      assertNull(jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      HashMap<String, ClientInfoStatus> hashMap0 = new HashMap<String, ClientInfoStatus>();
      SQLClientInfoException sQLClientInfoException0 = new SQLClientInfoException("", hashMap0);
      SQLRecoverableException sQLRecoverableException0 = new SQLRecoverableException("", "xKRN", sQLClientInfoException0);
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.serializerInstance((Annotated) null, sQLRecoverableException0);
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
      ObjectIdGenerators.UUIDGenerator objectIdGenerators_UUIDGenerator0 = new ObjectIdGenerators.UUIDGenerator();
      boolean boolean0 = defaultSerializerProvider_Impl0.includeFilterSuppressNulls(objectIdGenerators_UUIDGenerator0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      boolean boolean0 = defaultSerializerProvider_Impl0.includeFilterSuppressNulls((Object) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectIdGenerators.UUIDGenerator objectIdGenerators_UUIDGenerator0 = new ObjectIdGenerators.UUIDGenerator();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.findObjectId(objectIdGenerators_UUIDGenerator0, objectIdGenerators_UUIDGenerator0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      Class<ObjectIdGenerators.UUIDGenerator> class0 = ObjectIdGenerators.UUIDGenerator.class;
      boolean boolean0 = defaultSerializerProvider_Impl0.hasSerializerFor(class0, atomicReference0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      AtomicReference<Throwable> atomicReference0 = new AtomicReference<Throwable>();
      Class<Object> class0 = Object.class;
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
  public void test14()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<BeanSerializer> class0 = BeanSerializer.class;
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.serializeValue((JsonGenerator) null, (Object) class0);
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
      Class<CoreXMLSerializers.XMLGregorianCalendarSerializer> class0 = CoreXMLSerializers.XMLGregorianCalendarSerializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base(defaultSerializerProvider_Impl0);
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.acceptJsonFormatVisitor(simpleType0, jsonFormatVisitorWrapper_Base0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.acceptJsonFormatVisitor((JavaType) null, (JsonFormatVisitorWrapper) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // A class must be provided
         //
         verifyException("com.fasterxml.jackson.databind.ser.DefaultSerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      DefaultSerializerProvider defaultSerializerProvider0 = defaultSerializerProvider_Impl0.copy();
      assertNotSame(defaultSerializerProvider0, defaultSerializerProvider_Impl0);
  }
}
