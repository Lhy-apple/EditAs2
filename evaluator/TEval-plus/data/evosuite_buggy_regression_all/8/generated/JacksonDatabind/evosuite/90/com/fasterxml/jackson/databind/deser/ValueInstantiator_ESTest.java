/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:17:58 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.deser.impl.PropertyValueBuffer;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.node.BigIntegerNode;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.LinkedHashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ValueInstantiator_ESTest extends ValueInstantiator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      AnnotatedWithParams annotatedWithParams0 = valueInstantiator_Base0.getWithArgsCreator();
      assertNull(annotatedWithParams0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<String> class0 = String.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createUsingDefault((DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<JavaType> class0 = JavaType.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromInt((DeserializationContext) null, 255);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      boolean boolean0 = valueInstantiator_Base0.canInstantiate();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<BigIntegerNode> class0 = BigIntegerNode.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromDouble((DeserializationContext) null, (-2.147483648E9));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<JavaType> class0 = JavaType.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createUsingDelegate((DeserializationContext) null, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Integer> class0 = Integer.TYPE;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      AnnotatedWithParams annotatedWithParams0 = valueInstantiator_Base0.getDelegateCreator();
      assertNull(annotatedWithParams0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      JavaType javaType0 = valueInstantiator_Base0.getDelegateType((DeserializationConfig) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<SimpleObjectIdResolver> class0 = SimpleObjectIdResolver.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromObjectWith((DeserializationContext) null, (SettableBeanProperty[]) null, (PropertyValueBuffer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      JavaType javaType0 = valueInstantiator_Base0.getArrayDelegateType((DeserializationConfig) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      AnnotatedParameter annotatedParameter0 = valueInstantiator_Base0.getIncompleteParameter();
      assertNull(annotatedParameter0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      AnnotatedWithParams annotatedWithParams0 = valueInstantiator_Base0.getArrayDelegateCreator();
      assertNull(annotatedWithParams0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Integer> class0 = Integer.TYPE;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromString(defaultDeserializationContext_Impl0, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      Object[] objectArray0 = new Object[0];
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromObjectWith((DeserializationContext) null, objectArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createUsingArrayDelegate((DeserializationContext) null, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      SettableBeanProperty[] settableBeanPropertyArray0 = valueInstantiator_Base0.getFromObjectArguments((DeserializationConfig) null);
      assertNull(settableBeanPropertyArray0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedHashMap> class0 = LinkedHashMap.class;
      Class<Object> class1 = Object.class;
      Class<Integer> class2 = Integer.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class1, class2);
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(mapType0);
      boolean boolean0 = valueInstantiator_Base0.canCreateUsingArrayDelegate();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromBoolean((DeserializationContext) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromLong((DeserializationContext) null, 1169L);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      String string0 = valueInstantiator_Base0.getValueTypeDesc();
      assertEquals("java.lang.Integer", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromString((DeserializationContext) null, "E@/<<9m,^$hpjnb");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }
}
