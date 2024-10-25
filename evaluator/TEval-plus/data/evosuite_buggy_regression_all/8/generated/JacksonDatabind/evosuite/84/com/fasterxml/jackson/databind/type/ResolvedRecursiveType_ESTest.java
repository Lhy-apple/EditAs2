/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:16:32 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ResolvedRecursiveType_ESTest extends ResolvedRecursiveType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<NamedType> class0 = NamedType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ReferenceType referenceType0 = new ReferenceType(resolvedRecursiveType0, resolvedRecursiveType0);
      resolvedRecursiveType0.setReference(referenceType0);
      Class<String> class1 = String.class;
      ResolvedRecursiveType resolvedRecursiveType1 = new ResolvedRecursiveType(class1, typeBindings0);
      boolean boolean0 = resolvedRecursiveType0.equals(resolvedRecursiveType1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      // Undeclared exception!
      try { 
        resolvedRecursiveType0.getGenericSignature();
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
      Class<String> class0 = String.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      JavaType javaType0 = resolvedRecursiveType0.withTypeHandler(class0);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<String> class0 = String.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      JavaType javaType0 = resolvedRecursiveType0.withContentType(resolvedRecursiveType0);
      assertFalse(javaType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<String> class0 = String.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      boolean boolean0 = resolvedRecursiveType0.isContainerType();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<String> class0 = String.class;
      Class<SerializationFeature> class1 = SerializationFeature.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class1, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withContentTypeHandler((Object) null);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
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
      Class<String> class0 = String.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      JavaType javaType0 = resolvedRecursiveType0.withValueHandler(resolvedRecursiveType0);
      assertEquals(0, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType[] javaTypeArray0 = new JavaType[2];
      JavaType javaType0 = resolvedRecursiveType0.refine(class0, typeBindings0, (JavaType) null, javaTypeArray0);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<String> class0 = String.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      JavaType javaType0 = resolvedRecursiveType0.withStaticTyping();
      assertFalse(javaType0.isContainerType());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<SerializationFeature> class0 = SerializationFeature.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class0, (JavaType) simpleType0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withContentValueHandler(resolvedRecursiveType0);
      assertSame(resolvedRecursiveType0, javaType0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<String> class0 = String.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      JavaType javaType0 = resolvedRecursiveType0._narrow(class0);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<NamedType> class0 = NamedType.class;
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ReferenceType referenceType0 = new ReferenceType(resolvedRecursiveType0, resolvedRecursiveType0);
      resolvedRecursiveType0.setReference(referenceType0);
      // Undeclared exception!
      try { 
        resolvedRecursiveType0.setReference(referenceType0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Trying to re-set self reference; old value = [reference type, class com.fasterxml.jackson.databind.jsontype.NamedType<com.fasterxml.jackson.databind.jsontype.NamedType<[recursive type; com.fasterxml.jackson.databind.jsontype.NamedType>], new = [reference type, class com.fasterxml.jackson.databind.jsontype.NamedType<com.fasterxml.jackson.databind.jsontype.NamedType<[recursive type; com.fasterxml.jackson.databind.jsontype.NamedType>]
         //
         verifyException("com.fasterxml.jackson.databind.type.ResolvedRecursiveType", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      String string0 = resolvedRecursiveType0.getTypeName();
      assertEquals("[recursive type; UNRESOLVED", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<NamedType> class0 = NamedType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      boolean boolean0 = resolvedRecursiveType0.equals(resolvedRecursiveType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<NamedType> class0 = NamedType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      boolean boolean0 = resolvedRecursiveType0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<String> class0 = String.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      SerializationFeature serializationFeature0 = SerializationFeature.WRITE_CHAR_ARRAYS_AS_JSON_ARRAYS;
      boolean boolean0 = resolvedRecursiveType0.equals(serializationFeature0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<NamedType> class0 = NamedType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ReferenceType referenceType0 = new ReferenceType(resolvedRecursiveType0, resolvedRecursiveType0);
      resolvedRecursiveType0.setReference(referenceType0);
      boolean boolean0 = resolvedRecursiveType0.equals(referenceType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<NamedType> class0 = NamedType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ReferenceType referenceType0 = new ReferenceType(resolvedRecursiveType0, resolvedRecursiveType0);
      resolvedRecursiveType0.setReference(referenceType0);
      Class<Object> class1 = Object.class;
      ResolvedRecursiveType resolvedRecursiveType1 = new ResolvedRecursiveType(class1, typeBindings0);
      assertFalse(resolvedRecursiveType1.equals((Object)resolvedRecursiveType0));
      
      resolvedRecursiveType1.setReference(referenceType0);
      boolean boolean0 = resolvedRecursiveType0.equals(resolvedRecursiveType1);
      assertTrue(resolvedRecursiveType1.equals((Object)resolvedRecursiveType0));
      assertTrue(boolean0);
  }
}
