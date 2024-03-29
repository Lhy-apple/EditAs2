/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:57:29 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.databind.DatabindContext;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.io.IOException;
import java.lang.reflect.Type;
import java.math.BigDecimal;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DatabindContext_ESTest extends DatabindContext_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<String> class0 = String.class;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.reportBadDefinition((Class<?>) class0, (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JavaType javaType0 = defaultDeserializationContext_Impl0.constructType((Type) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      Class<Module> class1 = Module.class;
      // Undeclared exception!
      try { 
        deserializationContext0.constructSpecializedType(simpleType0, class1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DatabindContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JavaType javaType0 = deserializationContext0.constructSpecializedType(simpleType0, class0);
      assertFalse(javaType0.hasValueHandler());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      try { 
        defaultDeserializationContext_Impl0.resolveSubType((JavaType) null, "");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Could not resolve type id '' as a subtype of null: problem: (java.lang.NullPointerException) null
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidTypeIdException", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object[] objectArray0 = new Object[0];
      try { 
        defaultDeserializationContext_Impl0.reportInputMismatch((JavaType) null, "", objectArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // 
         //
         verifyException("com.fasterxml.jackson.databind.exc.MismatchedInputException", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<String> class0 = String.class;
      BigDecimal bigDecimal0 = new BigDecimal(2524L);
      Object[] objectArray0 = new Object[2];
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.handleWeirdNumberValue(class0, bigDecimal0, "JSON", objectArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      String string0 = ((DatabindContext)defaultSerializerProvider_Impl0)._desc("p[7eIKe,W");
      assertEquals("p[7eIKe,W", string0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Class<Integer> class0 = Integer.TYPE;
      JsonMappingException jsonMappingException0 = deserializationContext0.weirdKeyException(class0, "JSON", "NmH?du{%");
      assertNotNull(jsonMappingException0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<ObjectIdGenerators.UUIDGenerator> class0 = ObjectIdGenerators.UUIDGenerator.class;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.weirdStringException((String) null, class0, (String) null);
      assertNotNull(jsonMappingException0);
  }
}
