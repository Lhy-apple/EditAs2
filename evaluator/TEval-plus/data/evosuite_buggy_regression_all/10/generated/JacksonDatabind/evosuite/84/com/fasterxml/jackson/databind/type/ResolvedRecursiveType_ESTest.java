/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:38:26 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.ArrayList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ResolvedRecursiveType_ESTest extends ResolvedRecursiveType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ResolvedRecursiveType resolvedRecursiveType1 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType1.containedTypeOrUnknown(655);
      resolvedRecursiveType0.setReference(javaType0);
      boolean boolean0 = resolvedRecursiveType0.equals(resolvedRecursiveType1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      StringBuilder stringBuilder0 = new StringBuilder();
      // Undeclared exception!
      try { 
        resolvedRecursiveType0.getGenericSignature(stringBuilder0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.ResolvedRecursiveType", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withTypeHandler("");
      assertFalse(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withContentType((JavaType) null);
      assertTrue(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      boolean boolean0 = resolvedRecursiveType0.isContainerType();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withContentTypeHandler((Object) null);
      assertFalse(javaType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      // Undeclared exception!
      try { 
        resolvedRecursiveType0.getErasedSignature();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.ResolvedRecursiveType", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      JavaType javaType0 = resolvedRecursiveType0.withValueHandler((Object) null);
      assertFalse(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ReferenceType referenceType0 = new ReferenceType(resolvedRecursiveType0, resolvedRecursiveType0);
      JavaType[] javaTypeArray0 = new JavaType[4];
      JavaType javaType0 = resolvedRecursiveType0.refine(class0, typeBindings0, referenceType0, javaTypeArray0);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      JavaType javaType0 = resolvedRecursiveType0.withStaticTyping();
      assertSame(resolvedRecursiveType0, javaType0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class0, (JavaType) simpleType0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withContentValueHandler(resolvedRecursiveType0);
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      JavaType javaType0 = resolvedRecursiveType0._narrow(class0);
      assertSame(resolvedRecursiveType0, javaType0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      resolvedRecursiveType0.setReference(simpleType0);
      // Undeclared exception!
      try { 
        resolvedRecursiveType0.setReference(resolvedRecursiveType0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Trying to re-set self reference; old value = [simple type, class boolean], new = [recursive type; boolean
         //
         verifyException("com.fasterxml.jackson.databind.type.ResolvedRecursiveType", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ResolvedRecursiveType resolvedRecursiveType1 = new ResolvedRecursiveType(class0, typeBindings0);
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      resolvedRecursiveType0.setReference(simpleType0);
      // Undeclared exception!
      try { 
        resolvedRecursiveType0.setReference(resolvedRecursiveType1);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Trying to re-set self reference; old value = [simple type, class boolean], new = [recursive type; UNRESOLVED
         //
         verifyException("com.fasterxml.jackson.databind.type.ResolvedRecursiveType", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ResolvedRecursiveType resolvedRecursiveType1 = new ResolvedRecursiveType(class0, typeBindings0);
      resolvedRecursiveType1.setReference(resolvedRecursiveType0);
      assertFalse(resolvedRecursiveType1.equals((Object)resolvedRecursiveType0));
      
      resolvedRecursiveType0.setReference(resolvedRecursiveType0);
      boolean boolean0 = resolvedRecursiveType1.equals(resolvedRecursiveType0);
      assertTrue(resolvedRecursiveType1.equals((Object)resolvedRecursiveType0));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      boolean boolean0 = resolvedRecursiveType0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Integer> class0 = Integer.TYPE;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      Object object0 = new Object();
      boolean boolean0 = resolvedRecursiveType0.equals(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      resolvedRecursiveType0.setReference(simpleType0);
      boolean boolean0 = resolvedRecursiveType0.equals(simpleType0);
      assertFalse(boolean0);
  }
}
