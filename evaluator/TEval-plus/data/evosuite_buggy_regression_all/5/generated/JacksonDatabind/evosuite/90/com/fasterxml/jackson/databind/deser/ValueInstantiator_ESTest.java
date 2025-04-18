/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:07:54 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdReferenceProperty;
import com.fasterxml.jackson.databind.deser.impl.PropertyValueBuffer;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.node.DecimalNode;
import java.time.Month;
import java.time.chrono.ChronoLocalDate;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ValueInstantiator_ESTest extends ValueInstantiator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      AnnotatedWithParams annotatedWithParams0 = valueInstantiator_Base0.getWithArgsCreator();
      assertNull(annotatedWithParams0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Object> class0 = Object.class;
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
      Class<ObjectIdReferenceProperty> class0 = ObjectIdReferenceProperty.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromInt((DeserializationContext) null, 5);
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
      Class<DecimalNode> class0 = DecimalNode.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      boolean boolean0 = valueInstantiator_Base0.canInstantiate();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Month> class0 = Month.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromDouble((DeserializationContext) null, 0.0);
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
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createUsingDelegate((DeserializationContext) null, class0);
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
      Class<String> class0 = String.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      AnnotatedWithParams annotatedWithParams0 = valueInstantiator_Base0.getDelegateCreator();
      assertNull(annotatedWithParams0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<String> class0 = String.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      JavaType javaType0 = valueInstantiator_Base0.getDelegateType((DeserializationConfig) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<DecimalNode> class0 = DecimalNode.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory((DeserializerFactoryConfig) null);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromObjectWith((DeserializationContext) defaultDeserializationContext_Impl0, (SettableBeanProperty[]) null, (PropertyValueBuffer) null);
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
      Class<DecimalNode> class0 = DecimalNode.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromString((DeserializationContext) null, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromObjectWith((DeserializationContext) null, (Object[]) null);
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
      Class<String> class0 = String.class;
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
      Class<Month> class0 = Month.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      SettableBeanProperty[] settableBeanPropertyArray0 = valueInstantiator_Base0.getFromObjectArguments((DeserializationConfig) null);
      assertNull(settableBeanPropertyArray0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<String> class0 = String.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      boolean boolean0 = valueInstantiator_Base0.canCreateUsingArrayDelegate();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<Month> class0 = Month.class;
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
      Class<Object> class0 = Object.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0.createFromLong((DeserializationContext) null, 630L);
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
      Class<Object> class0 = Object.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      String string0 = valueInstantiator_Base0.getValueTypeDesc();
      assertEquals("java.lang.Object", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ValueInstantiator.Base valueInstantiator_Base0 = null;
      try {
        valueInstantiator_Base0 = new ValueInstantiator.Base((JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator$Base", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<String> class0 = String.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0._createFromStringFallbacks((DeserializationContext) null, "#@ iz^YQ+9i+");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      // Undeclared exception!
      try { 
        valueInstantiator_Base0._createFromStringFallbacks(defaultDeserializationContext_Impl0, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }
}
