/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:19:35 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DatabindContext;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.introspect.Annotated;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.jsontype.impl.TypeNameIdResolver;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.util.Converter;
import java.io.IOException;
import java.lang.reflect.Type;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DatabindContext_ESTest extends DatabindContext_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<SimpleObjectIdResolver> class0 = SimpleObjectIdResolver.class;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.reportBadDefinition((Class<?>) class0, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      JavaType javaType0 = serializerProvider0.constructType((Type) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      Class<LongNode> class0 = LongNode.class;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.constructSpecializedType(javaType0, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DatabindContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Class<TypeNameIdResolver> class0 = TypeNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JavaType javaType0 = deserializationContext0.constructSpecializedType(simpleType0, class0);
      assertSame(simpleType0, javaType0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      try { 
        defaultSerializerProvider_Impl0.resolveSubType((JavaType) null, "KR t$&{o&3krE");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Could not resolve type id 'KR t$&{o&3krE' as a subtype of null: problem: (java.lang.NullPointerException) null
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidTypeIdException", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.resolveSubType((JavaType) null, "; xpected type Converter or Class<Converter> instead");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      // Undeclared exception!
      try { 
        defaultSerializerProvider_Impl0.converterInstance((Annotated) null, simpleObjectIdResolver0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // AnnotationIntrospector returned Converter definition of type com.fasterxml.jackson.annotation.SimpleObjectIdResolver; expected type Converter or Class<Converter> instead
         //
         verifyException("com.fasterxml.jackson.databind.DatabindContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Converter<Object, Object> converter0 = defaultSerializerProvider_Impl0.converterInstance((Annotated) null, (Object) null);
      assertNull(converter0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Class<BasicBeanDescription> class0 = BasicBeanDescription.class;
      Object[] objectArray0 = new Object[0];
      // Undeclared exception!
      try { 
        deserializationContext0.handleWeirdStringValue(class0, "f51{,{Vl]", "f51{,{Vl]", objectArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Class<Integer> class0 = Integer.class;
      Object[] objectArray0 = new Object[5];
      // Undeclared exception!
      try { 
        deserializationContext0.handleWeirdStringValue(class0, "B", "LS", objectArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = (DefaultDeserializationContext.Impl)objectMapper0.getDeserializationContext();
      String string0 = ((DatabindContext)defaultDeserializationContext_Impl0)._desc("f51{,{Vl]");
      assertEquals("f51{,{Vl]", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = (DefaultDeserializationContext.Impl)objectMapper0.getDeserializationContext();
      String string0 = ((DatabindContext)defaultDeserializationContext_Impl0)._quotedString("B");
      assertEquals("\"B\"", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = (DefaultDeserializationContext.Impl)objectMapper0.getDeserializationContext();
      String string0 = ((DatabindContext)defaultDeserializationContext_Impl0)._quotedString((String) null);
      assertEquals("[N/A]", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper((JsonFactory) null);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = (DefaultDeserializationContext.Impl)objectMapper0.getDeserializationContext();
      String string0 = ((DatabindContext)defaultDeserializationContext_Impl0)._desc((String) null);
      assertEquals("[N/A]", string0);
  }
}
