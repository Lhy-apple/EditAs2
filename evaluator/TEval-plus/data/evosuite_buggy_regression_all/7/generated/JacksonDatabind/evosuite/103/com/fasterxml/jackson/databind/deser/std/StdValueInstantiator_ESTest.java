/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:14:26 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.StdValueInstantiator;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.node.DoubleNode;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Type;
import java.sql.SQLNonTransientException;
import java.sql.SQLRecoverableException;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdValueInstantiator_ESTest extends StdValueInstantiator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<JsonFactory.Feature> class0 = JsonFactory.Feature.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getDelegateType((DeserializationConfig) null);
      assertEquals("`com.fasterxml.jackson.core.JsonFactory$Feature`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.configureIncompleteParameter((AnnotatedParameter) null);
      assertEquals("`java.lang.reflect.InvocationTargetException`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<JsonMappingException> class0 = JsonMappingException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createUsingArrayDelegate(defaultDeserializationContext_Impl0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No delegate constructor for `com.fasterxml.jackson.databind.JsonMappingException`
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Object> class0 = Object.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getDelegateCreator();
      assertEquals("`java.lang.Object`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Object> class0 = Object.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.configureFromBooleanCreator((AnnotatedWithParams) null);
      assertEquals("`java.lang.Object`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<JsonFactory.Feature> class0 = JsonFactory.Feature.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getArrayDelegateCreator();
      assertEquals("`com.fasterxml.jackson.core.JsonFactory$Feature`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Object> class0 = Object.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.configureFromStringCreator((AnnotatedWithParams) null);
      assertEquals("`java.lang.Object`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Object> class0 = Object.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.configureFromIntCreator((AnnotatedWithParams) null);
      assertEquals("`java.lang.Object`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<JsonFactory.Feature> class0 = JsonFactory.Feature.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.configureFromDoubleCreator((AnnotatedWithParams) null);
      assertEquals("`com.fasterxml.jackson.core.JsonFactory$Feature`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Boolean> class0 = Boolean.TYPE;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      JavaType javaType0 = typeFactory0.constructType((Type) class0, typeBindings0);
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, javaType0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromInt(defaultDeserializationContext_Impl0, 0);
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
      Class<Integer> class0 = Integer.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getFromObjectArguments((DeserializationConfig) null);
      assertEquals("`java.lang.Integer`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Boolean> class0 = Boolean.TYPE;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      JavaType javaType0 = typeFactory0.constructType((Type) class0, typeBindings0);
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, javaType0);
      StdValueInstantiator stdValueInstantiator1 = new StdValueInstantiator(stdValueInstantiator0);
      assertEquals("[simple type, class boolean]", stdValueInstantiator1.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<DoubleNode> class0 = DoubleNode.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getArrayDelegateType((DeserializationConfig) null);
      assertEquals("`com.fasterxml.jackson.databind.node.DoubleNode`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<BasicBeanDescription> class0 = BasicBeanDescription.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getDefaultCreator();
      assertEquals("`com.fasterxml.jackson.databind.introspect.BasicBeanDescription`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Object> class0 = Object.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getIncompleteParameter();
      assertEquals("`java.lang.Object`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<JsonFactory.Feature> class0 = JsonFactory.Feature.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.getWithArgsCreator();
      assertEquals("`com.fasterxml.jackson.core.JsonFactory$Feature`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      stdValueInstantiator0.configureFromLongCreator((AnnotatedWithParams) null);
      assertEquals("`java.lang.Integer`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, (Class<?>) null);
      assertFalse(stdValueInstantiator0.canCreateFromString());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, (JavaType) null);
      assertEquals("UNKNOWN TYPE", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<JsonMappingException> class0 = JsonMappingException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      boolean boolean0 = stdValueInstantiator0.canInstantiate();
      assertFalse(boolean0);
      assertEquals("`com.fasterxml.jackson.databind.JsonMappingException`", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<BasicBeanDescription> class0 = BasicBeanDescription.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createUsingDefault(defaultDeserializationContext_Impl0);
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
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<JsonMappingException> class0 = JsonMappingException.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
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
  public void test22()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<SimpleType> class0 = SimpleType.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createUsingDelegate(defaultDeserializationContext_Impl0, beanDeserializerFactory0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No delegate constructor for `com.fasterxml.jackson.databind.type.SimpleType`
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdValueInstantiator", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<SimpleType> class0 = SimpleType.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromString(defaultDeserializationContext_Impl0, "No Bindings!");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Object> class0 = Object.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromLong(defaultDeserializationContext_Impl0, 0L);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Object> class0 = Object.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromDouble(defaultDeserializationContext_Impl0, 1193.0676366369514);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Object> class0 = Object.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.createFromBoolean(defaultDeserializationContext_Impl0, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<BasicBeanDescription> class0 = BasicBeanDescription.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      SQLNonTransientException sQLNonTransientException0 = new SQLNonTransientException((String) null, (String) null);
      JsonMappingException jsonMappingException0 = stdValueInstantiator0.wrapException(sQLNonTransientException0);
      SQLRecoverableException sQLRecoverableException0 = new SQLRecoverableException(jsonMappingException0);
      JsonMappingException jsonMappingException1 = stdValueInstantiator0.wrapException(sQLRecoverableException0);
      assertSame(jsonMappingException1, jsonMappingException0);
      assertEquals("java.sql.SQLRecoverableException: com.fasterxml.jackson.databind.JsonMappingException: Instantiation of `com.fasterxml.jackson.databind.introspect.BasicBeanDescription` value failed: null", sQLRecoverableException0.toString());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Boolean> class0 = Boolean.TYPE;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      JavaType javaType0 = typeFactory0.constructType((Type) class0, typeBindings0);
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, javaType0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      SQLRecoverableException sQLRecoverableException0 = new SQLRecoverableException("$/", "delegate");
      // Undeclared exception!
      try { 
        stdValueInstantiator0.unwrapAndWrapException(defaultDeserializationContext_Impl0, sQLRecoverableException0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Boolean> class0 = Boolean.TYPE;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      JavaType javaType0 = typeFactory0.constructType((Type) class0, typeBindings0);
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, javaType0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.weirdStringException((String) null, class0, (String) null);
      stdValueInstantiator0.unwrapAndWrapException(defaultDeserializationContext_Impl0, jsonMappingException0);
      assertEquals("[simple type, class boolean]", stdValueInstantiator0.getValueTypeDesc());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Object> class0 = Object.class;
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, class0);
      // Undeclared exception!
      try { 
        stdValueInstantiator0.rewrapCtorProblem(defaultDeserializationContext_Impl0, (Throwable) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      StdValueInstantiator stdValueInstantiator0 = new StdValueInstantiator((DeserializationConfig) null, mapType0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.missingTypeIdException(mapType0, "No delegate constructor for ");
      stdValueInstantiator0.rewrapCtorProblem(defaultDeserializationContext_Impl0, jsonMappingException0);
      assertEquals("[map type; class java.util.HashMap, [simple type, class java.lang.Object] -> [simple type, class java.lang.Object]]", stdValueInstantiator0.getValueTypeDesc());
  }
}
