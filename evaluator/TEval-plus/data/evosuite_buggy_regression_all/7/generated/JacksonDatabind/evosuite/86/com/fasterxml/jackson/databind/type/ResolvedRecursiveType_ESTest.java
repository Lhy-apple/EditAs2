/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:08:24 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.LRUMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ResolvedRecursiveType_ESTest extends ResolvedRecursiveType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      resolvedRecursiveType0.setReference(resolvedRecursiveType0);
      ResolvedRecursiveType resolvedRecursiveType1 = new ResolvedRecursiveType(class0, typeBindings0);
      boolean boolean0 = resolvedRecursiveType0.equals(resolvedRecursiveType1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Vector<JavaType> vector0 = new Vector<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) vector0);
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
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withTypeHandler(typeBindings0);
      assertFalse(javaType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      LRUMap<Object, JavaType> lRUMap0 = new LRUMap<Object, JavaType>(1530, 1530);
      TypeFactory typeFactory0 = new TypeFactory(lRUMap0);
      Class<ArrayList> class1 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class1);
      JavaType javaType0 = resolvedRecursiveType0.withContentType(collectionType0);
      assertFalse(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      boolean boolean0 = resolvedRecursiveType0.isContainerType();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withContentTypeHandler(resolvedRecursiveType0);
      assertEquals(0, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<String> class0 = String.class;
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
      JavaType javaType0 = resolvedRecursiveType0.withValueHandler(class0);
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      Class<Object> class1 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[8];
      JavaType javaType0 = resolvedRecursiveType0.refine(class1, typeBindings0, javaTypeArray0[4], javaTypeArray0);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withStaticTyping();
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Vector<JavaType> vector0 = new Vector<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) vector0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0.withContentValueHandler(typeBindings0);
      assertTrue(javaType0.hasContentType());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType javaType0 = resolvedRecursiveType0._narrow(class0);
      assertFalse(javaType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      resolvedRecursiveType0.setReference(resolvedRecursiveType0);
      // Undeclared exception!
      try { 
        resolvedRecursiveType0.setReference(resolvedRecursiveType0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Trying to re-set self reference; old value = [recursive type; java.lang.Integer, new = [recursive type; java.lang.Integer
         //
         verifyException("com.fasterxml.jackson.databind.type.ResolvedRecursiveType", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      String string0 = resolvedRecursiveType0.toString();
      assertEquals("[recursive type; UNRESOLVED", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      Class<Object> class1 = Object.class;
      ResolvedRecursiveType resolvedRecursiveType1 = new ResolvedRecursiveType(class1, typeBindings0);
      boolean boolean0 = resolvedRecursiveType0.equals(resolvedRecursiveType1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      resolvedRecursiveType0.setReference(resolvedRecursiveType0);
      MapperFeature mapperFeature0 = MapperFeature.INFER_PROPERTY_MUTATORS;
      boolean boolean0 = resolvedRecursiveType0.equals(mapperFeature0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      resolvedRecursiveType0.setReference(resolvedRecursiveType0);
      Class<Object> class1 = Object.class;
      ResolvedRecursiveType resolvedRecursiveType1 = new ResolvedRecursiveType(class1, typeBindings0);
      assertFalse(resolvedRecursiveType1.equals((Object)resolvedRecursiveType0));
      
      resolvedRecursiveType1.setReference(resolvedRecursiveType0);
      boolean boolean0 = resolvedRecursiveType0.equals(resolvedRecursiveType1);
      assertTrue(resolvedRecursiveType1.equals((Object)resolvedRecursiveType0));
      assertTrue(boolean0);
  }
}