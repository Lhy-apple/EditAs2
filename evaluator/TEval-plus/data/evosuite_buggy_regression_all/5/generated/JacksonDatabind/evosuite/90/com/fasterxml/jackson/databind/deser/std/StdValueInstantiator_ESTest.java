/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:08:57 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.std.StdValueInstantiator;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.fasterxml.jackson.databind.node.ShortNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.sql.SQLTransactionRollbackException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.lang.MockThrowable;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdValueInstantiator_ESTest extends StdValueInstantiator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<ShortNode> class0 = ShortNode.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getDelegateType((DeserializationConfig) null);
      assertEquals("com.fasterxml.jackson.databind.node.ShortNode", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ExceptionInInitializerError exceptionInInitializerError0 = new ExceptionInInitializerError(" value failed: ");
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(exceptionInInitializerError0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<NamedType> class0 = NamedType.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createUsingArrayDelegate((DeserializationContext) null, (Object) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No delegate constructor for com.fasterxml.jackson.databind.jsontype.NamedType
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getDelegateCreator();
      assertEquals("java.lang.reflect.InvocationTargetException", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getArrayDelegateCreator();
      assertEquals("java.lang.reflect.InvocationTargetException", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<NamedType> class0 = NamedType.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromInt((DeserializationContext) null, 2810);
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
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      StdValueInstantiator stdValueInstantiator1 = new StdValueInstantiator(stdValueInstantiator0);
      assertEquals("java.lang.reflect.InvocationTargetException", stdValueInstantiator1.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<JsonMappingException> class0 = JsonMappingException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getArrayDelegateType((DeserializationConfig) null);
      assertEquals("com.fasterxml.jackson.databind.JsonMappingException", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<ShortNode> class0 = ShortNode.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getDefaultCreator();
      assertEquals("com.fasterxml.jackson.databind.node.ShortNode", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<ObjectReader> class0 = ObjectReader.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getWithArgsCreator();
      assertEquals("com.fasterxml.jackson.databind.ObjectReader", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, (Class<?>) null);
      assertEquals("UNKNOWN TYPE", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, (JavaType) null);
      assertEquals("UNKNOWN TYPE", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<ReferenceType> class0 = ReferenceType.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      boolean boolean0 = stdValueInstantiator0.canInstantiate();
      assertEquals("com.fasterxml.jackson.databind.type.ReferenceType", stdValueInstantiator0.getValueTypeDesc());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<SettableBeanProperty> class0 = SettableBeanProperty.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      Class<Integer> class1 = Integer.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class1, (TypeBindings) null);
      stdValueInstantiator0._arrayDelegateType = (JavaType) resolvedRecursiveType0;
      boolean boolean0 = stdValueInstantiator0.canCreateUsingArrayDelegate();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<ShortNode> class0 = ShortNode.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createUsingDefault((DeserializationContext) null);
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
      Class<ExceptionInInitializerError> class0 = ExceptionInInitializerError.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromObjectWith((DeserializationContext) null, (Object[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createUsingDelegate((DeserializationContext) null, (Object) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No delegate constructor for java.lang.Integer
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<DeserializationFeature> class0 = DeserializationFeature.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromString((DeserializationContext) null, "No delegate constructor for ");
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
      Class<BasicBeanDescription> class0 = BasicBeanDescription.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromLong((DeserializationContext) null, (-362));
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
      Class<JsonMappingException> class0 = JsonMappingException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromDouble((DeserializationContext) null, 0.0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<JsonMappingException> class0 = JsonMappingException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromBoolean((DeserializationContext) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<DeserializationFeature> class0 = DeserializationFeature.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      MockThrowable mockThrowable0 = new MockThrowable("X$7>NgbnR({ii(!");
      stdValueInstantiator0.wrapException(mockThrowable0);
      assertEquals("com.fasterxml.jackson.databind.DeserializationFeature", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<DeserializationFeature> class0 = DeserializationFeature.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonMappingException jsonMappingException0 = JsonMappingException.from((SerializerProvider) defaultSerializerProvider_Impl0, (String) null);
      JsonMappingException jsonMappingException1 = stdValueInstantiator0.wrapException(jsonMappingException0);
      assertSame(jsonMappingException1, jsonMappingException0);
      assertEquals("com.fasterxml.jackson.databind.DeserializationFeature", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<InputStream> class0 = InputStream.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonMappingException jsonMappingException0 = JsonMappingException.from((SerializerProvider) defaultSerializerProvider_Impl0, "UNKNOWN TYPE");
      stdValueInstantiator0.unwrapAndWrapException((DeserializationContext) null, jsonMappingException0);
      assertEquals("java.io.InputStream", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<SettableBeanProperty> class0 = SettableBeanProperty.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      SQLTransactionRollbackException sQLTransactionRollbackException0 = new SQLTransactionRollbackException();
      // Undeclared exception!
      try { 
        stdValueInstantiator0.unwrapAndWrapException((DeserializationContext) null, sQLTransactionRollbackException0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<DeserializationFeature> class0 = DeserializationFeature.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonMappingException jsonMappingException0 = defaultSerializerProvider_Impl0.mappingException("jacisPRsn>MCmG{", (Object[]) null);
      stdValueInstantiator0.rewrapCtorProblem((DeserializationContext) null, jsonMappingException0);
      assertEquals("com.fasterxml.jackson.databind.DeserializationFeature", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<JsonFormat.Feature> class0 = JsonFormat.Feature.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      ExceptionInInitializerError exceptionInInitializerError0 = new ExceptionInInitializerError();
      // Undeclared exception!
      try { 
        stdValueInstantiator0.rewrapCtorProblem((DeserializationContext) null, exceptionInInitializerError0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<JsonFormat.Feature> class0 = JsonFormat.Feature.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      ExceptionInInitializerError exceptionInInitializerError0 = new ExceptionInInitializerError();
      ExceptionInInitializerError exceptionInInitializerError1 = new ExceptionInInitializerError(exceptionInInitializerError0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.rewrapCtorProblem((DeserializationContext) null, exceptionInInitializerError1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdValueInstantiator", e);
      }
  }
}