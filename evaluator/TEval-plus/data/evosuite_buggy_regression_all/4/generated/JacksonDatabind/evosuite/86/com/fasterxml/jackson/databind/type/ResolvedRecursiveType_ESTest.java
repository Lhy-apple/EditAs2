/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:45:43 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ResolvedRecursiveType_ESTest extends ResolvedRecursiveType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, (JavaType) null);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      // Undeclared exception!
      try { 
        resolvedRecursiveType0.getGenericSignature((StringBuilder) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.ResolvedRecursiveType", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      JavaType javaType0 = resolvedRecursiveType0.withTypeHandler(resolvedRecursiveType0);
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      TypeFactory typeFactory0 = TypeFactory.instance;
      ArrayType arrayType0 = typeFactory0.constructArrayType((JavaType) resolvedRecursiveType0);
      JavaType javaType0 = resolvedRecursiveType0.withContentType(arrayType0);
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      boolean boolean0 = resolvedRecursiveType0.isContainerType();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withContentTypeHandler("[recursive type; UNRESOLVED");
      assertFalse(javaType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      StringBuilder stringBuilder0 = new StringBuilder("");
      // Undeclared exception!
      try { 
        resolvedRecursiveType0.getErasedSignature(stringBuilder0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.ResolvedRecursiveType", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      TypeFactory typeFactory0 = TypeFactory.instance;
      ArrayType arrayType0 = typeFactory0.constructArrayType((JavaType) resolvedRecursiveType0);
      ArrayType arrayType1 = arrayType0.withContentValueHandler(typeFactory0);
      assertFalse(arrayType1.isAbstract());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      JavaType[] javaTypeArray0 = new JavaType[4];
      JavaType javaType0 = resolvedRecursiveType0.refine(class0, (TypeBindings) null, resolvedRecursiveType0, javaTypeArray0);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withStaticTyping();
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withContentValueHandler("[recursive type; UNRESOLVED");
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0._narrow(class0);
      assertFalse(javaType0.isContainerType());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      resolvedRecursiveType0.setReference(resolvedRecursiveType0);
      // Undeclared exception!
      try { 
        resolvedRecursiveType0.setReference(resolvedRecursiveType0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Trying to re-set self reference; old value = [recursive type; java.lang.String, new = [recursive type; java.lang.String
         //
         verifyException("com.fasterxml.jackson.databind.type.ResolvedRecursiveType", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      String string0 = resolvedRecursiveType0.toString();
      assertEquals("[recursive type; UNRESOLVED", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      boolean boolean0 = resolvedRecursiveType0.equals(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<Integer> class1 = Integer.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      resolvedRecursiveType0.setReference(resolvedRecursiveType0);
      ResolvedRecursiveType resolvedRecursiveType1 = new ResolvedRecursiveType(class1, typeBindings0);
      assertFalse(resolvedRecursiveType1.equals((Object)resolvedRecursiveType0));
      
      resolvedRecursiveType1.setReference(resolvedRecursiveType0);
      boolean boolean0 = resolvedRecursiveType1.equals(resolvedRecursiveType0);
      assertTrue(resolvedRecursiveType1.equals((Object)resolvedRecursiveType0));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      resolvedRecursiveType0.setReference(resolvedRecursiveType0);
      Class<ObjectMapper.DefaultTyping> class1 = ObjectMapper.DefaultTyping.class;
      ResolvedRecursiveType resolvedRecursiveType1 = new ResolvedRecursiveType(class1, typeBindings0);
      boolean boolean0 = resolvedRecursiveType0.equals(resolvedRecursiveType1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      resolvedRecursiveType0.setReference(resolvedRecursiveType0);
      boolean boolean0 = resolvedRecursiveType0.equals(typeBindings0);
      assertFalse(boolean0);
  }
}