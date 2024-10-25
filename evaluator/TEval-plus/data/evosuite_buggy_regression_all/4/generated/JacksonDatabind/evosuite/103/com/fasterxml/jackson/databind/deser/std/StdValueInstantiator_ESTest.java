/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:50:23 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.std.StdValueInstantiator;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.node.BigIntegerNode;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import java.io.BufferedInputStream;
import java.io.PipedInputStream;
import java.io.PushbackInputStream;
import java.lang.reflect.InvocationTargetException;
import java.sql.ClientInfoStatus;
import java.sql.SQLClientInfoException;
import java.sql.SQLTransactionRollbackException;
import java.util.HashMap;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdValueInstantiator_ESTest extends StdValueInstantiator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<ExceptionInInitializerError> class0 = ExceptionInInitializerError.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getDelegateType((DeserializationConfig) null);
      assertEquals("`java.lang.ExceptionInInitializerError`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<BigIntegerNode> class0 = BigIntegerNode.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, resolvedRecursiveType0);
      stdValueInstantiator0.configureIncompleteParameter((AnnotatedParameter) null);
      assertEquals("[recursive type; UNRESOLVED", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createUsingArrayDelegate((DeserializationContext) null, (Object) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No delegate constructor for `java.lang.reflect.InvocationTargetException`
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<ExceptionInInitializerError> class0 = ExceptionInInitializerError.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getDelegateCreator();
      assertEquals("`java.lang.ExceptionInInitializerError`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<ExceptionInInitializerError> class0 = ExceptionInInitializerError.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.configureFromArraySettings((AnnotatedWithParams) null, (JavaType) null, (SettableBeanProperty[]) null);
      assertEquals("`java.lang.ExceptionInInitializerError`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, (JavaType) null);
      stdValueInstantiator0.configureFromBooleanCreator((AnnotatedWithParams) null);
      assertEquals("UNKNOWN TYPE", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<ExceptionInInitializerError> class0 = ExceptionInInitializerError.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getArrayDelegateCreator();
      assertEquals("`java.lang.ExceptionInInitializerError`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.configureFromStringCreator((AnnotatedWithParams) null);
      assertEquals("`com.fasterxml.jackson.annotation.PropertyAccessor`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.configureFromIntCreator((AnnotatedWithParams) null);
      assertEquals("`java.lang.reflect.InvocationTargetException`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.configureFromDoubleCreator((AnnotatedWithParams) null);
      assertEquals("`com.fasterxml.jackson.annotation.ObjectIdResolver`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Object> class0 = Object.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      HashMap<String, ClientInfoStatus> hashMap0 = new HashMap<String, ClientInfoStatus>();
      SQLClientInfoException sQLClientInfoException0 = new SQLClientInfoException("", "q$EyI{LJ}to)\"B?$|-<", (-87), hashMap0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.unwrapAndWrapException((DeserializationContext) null, sQLClientInfoException0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<ExceptionInInitializerError> class0 = ExceptionInInitializerError.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getFromObjectArguments((DeserializationConfig) null);
      assertEquals("`java.lang.ExceptionInInitializerError`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      StdValueInstantiator stdValueInstantiator1 = new StdValueInstantiator(stdValueInstantiator0);
      assertEquals("`java.lang.reflect.InvocationTargetException`", stdValueInstantiator1.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<ExceptionInInitializerError> class0 = ExceptionInInitializerError.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getArrayDelegateType((DeserializationConfig) null);
      assertEquals("`java.lang.ExceptionInInitializerError`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getDefaultCreator();
      assertEquals("`com.fasterxml.jackson.annotation.PropertyAccessor`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<BigIntegerNode> class0 = BigIntegerNode.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, resolvedRecursiveType0);
      stdValueInstantiator0.getIncompleteParameter();
      assertEquals("[recursive type; UNRESOLVED", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getWithArgsCreator();
      assertEquals("`com.fasterxml.jackson.annotation.ObjectIdResolver`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, (JavaType) null);
      stdValueInstantiator0.configureFromLongCreator((AnnotatedWithParams) null);
      assertEquals("UNKNOWN TYPE", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      boolean boolean0 = stdValueInstantiator0.canInstantiate();
      assertEquals("`java.lang.reflect.InvocationTargetException`", stdValueInstantiator0.getValueTypeDesc());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
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
  public void test20()  throws Throwable  {
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromObjectWith((DeserializationContext) defaultDeserializationContext_Impl0, (Object[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<ExceptionInInitializerError> class0 = ExceptionInInitializerError.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory((DeserializerFactoryConfig) null);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createUsingDelegate(defaultDeserializationContext_Impl0, (Object) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No delegate constructor for `java.lang.ExceptionInInitializerError`
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromString((DeserializationContext) null, "k!X1#");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromInt((DeserializationContext) null, (-2495));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromLong((DeserializationContext) null, 0L);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.ValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<PropertyAccessor> class0 = PropertyAccessor.class;
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
  public void test26()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
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
  public void test27()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      SQLClientInfoException sQLClientInfoException0 = new SQLClientInfoException("zEl{,Y3+amMSj", "Value returned by 'any-getter' (%s()) not java.util.Map but %s", (-61), (Map<String, ClientInfoStatus>) null);
      stdValueInstantiator0.wrapException(sQLClientInfoException0);
      assertEquals("`java.lang.reflect.InvocationTargetException`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<Module> class0 = Module.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(pipedInputStream0, 80);
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(bufferedInputStream0);
      JsonMappingException jsonMappingException0 = new JsonMappingException(pushbackInputStream0, "U'B'~L8{dp%Gl5Z\u0002L");
      JsonMappingException jsonMappingException1 = stdValueInstantiator0.wrapException(jsonMappingException0);
      assertSame(jsonMappingException1, jsonMappingException0);
      assertEquals("`com.fasterxml.jackson.databind.Module`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, (JavaType) null);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.invalidTypeIdException((JavaType) null, "XfypFzcMZ|-p!R\"7", "XfypFzcMZ|-p!R\"7");
      stdValueInstantiator0.unwrapAndWrapException(defaultDeserializationContext_Impl0, jsonMappingException0);
      assertEquals("UNKNOWN TYPE", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      SQLTransactionRollbackException sQLTransactionRollbackException0 = new SQLTransactionRollbackException((String) null, (String) null);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.rewrapCtorProblem((DeserializationContext) null, sQLTransactionRollbackException0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdValueInstantiator", e);
      }
  }
}
