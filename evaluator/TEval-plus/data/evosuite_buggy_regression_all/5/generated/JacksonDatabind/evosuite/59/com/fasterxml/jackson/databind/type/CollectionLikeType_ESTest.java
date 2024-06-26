/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:04:30 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.LRUMap;
import java.util.Collection;
import java.util.LinkedList;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CollectionLikeType_ESTest extends CollectionLikeType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<MapType> class0 = MapType.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      String string0 = collectionType0.getErasedSignature();
      assertEquals("Ljava/util/LinkedList;", string0);
      assertFalse(collectionType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      Class<Integer> class1 = Integer.class;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class1, (JavaType) collectionType0);
      String string0 = collectionLikeType0.toString();
      assertFalse(collectionLikeType0.useStaticType());
      assertEquals("[collection-like type; class java.lang.Integer, contains [collection type; class java.util.LinkedList, contains [simple type, class java.lang.Object]]]", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Class<ArrayType> class1 = ArrayType.class;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class1, (JavaType) collectionType0);
      CollectionLikeType collectionLikeType1 = collectionLikeType0.withContentTypeHandler(collectionType0);
      assertFalse(collectionLikeType1.useStaticType());
      assertFalse(collectionLikeType0.hasHandlers());
      assertTrue(collectionLikeType1.equals((Object)collectionLikeType0));
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LRUMap<Object, JavaType> lRUMap0 = new LRUMap<Object, JavaType>(0, (-3167));
      TypeFactory typeFactory0 = new TypeFactory(lRUMap0);
      Class<LinkedList> class0 = LinkedList.class;
      Class<CollectionType> class1 = CollectionType.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      collectionType0.getContentValueHandler();
      assertFalse(collectionType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      collectionType0.getContentTypeHandler();
      assertFalse(collectionType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<ReferenceType> class0 = ReferenceType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class0, (JavaType) resolvedRecursiveType0);
      JavaType[] javaTypeArray0 = new JavaType[1];
      JavaType javaType0 = collectionLikeType0.refine(class0, typeBindings0, resolvedRecursiveType0, javaTypeArray0);
      assertTrue(javaType0.equals((Object)collectionLikeType0));
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(collectionType0, (JavaType) null);
      String string0 = collectionLikeType0.buildCanonicalName();
      assertFalse(collectionType0.hasHandlers());
      assertEquals("java.util.LinkedList", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      // Undeclared exception!
      try { 
        collectionType0.getGenericSignature((StringBuilder) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<MapType> class0 = MapType.class;
      SimpleType simpleType0 = new SimpleType(class0);
      CollectionLikeType collectionLikeType0 = new CollectionLikeType(simpleType0, simpleType0);
      Object object0 = new Object();
      CollectionLikeType collectionLikeType1 = collectionLikeType0.withTypeHandler(object0);
      assertFalse(collectionLikeType1.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<MapType> class0 = MapType.class;
      SimpleType simpleType0 = new SimpleType(class0);
      CollectionLikeType collectionLikeType0 = new CollectionLikeType(simpleType0, simpleType0);
      CollectionLikeType collectionLikeType1 = collectionLikeType0.withValueHandler(simpleType0);
      assertTrue(collectionLikeType1.isConcrete());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      Class<String> class1 = String.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class1, (JavaType) null);
      JavaType[] javaTypeArray0 = new JavaType[0];
      // Undeclared exception!
      try { 
        CollectionLikeType.construct(class0, typeBindings0, (JavaType) null, javaTypeArray0, (JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.CollectionLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      Class<ArrayType> class1 = ArrayType.class;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(collectionType0, collectionType0);
      CollectionLikeType collectionLikeType1 = collectionLikeType0.withContentValueHandler(class1);
      assertTrue(collectionLikeType1.equals((Object)collectionLikeType0));
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class0, (JavaType) simpleType0);
      JavaType javaType0 = collectionLikeType0._narrow(class0);
      assertTrue(javaType0.equals((Object)collectionLikeType0));
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class0, (JavaType) collectionType0);
      assertEquals(1, collectionLikeType0.containedTypeCount());
      assertFalse(collectionLikeType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      // Undeclared exception!
      try { 
        CollectionLikeType.upgradeFrom((JavaType) null, (JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.CollectionLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      Class<Integer> class1 = Integer.class;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class1, (JavaType) collectionType0);
      JavaType javaType0 = collectionLikeType0.withContentType(collectionLikeType0);
      assertNotSame(javaType0, collectionLikeType0);
      assertFalse(javaType0.equals((Object)collectionLikeType0));
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(javaType0, javaType0);
      JavaType javaType1 = collectionLikeType0.withContentType(javaType0);
      assertSame(javaType1, collectionLikeType0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class0, (JavaType) simpleType0);
      CollectionLikeType collectionLikeType1 = collectionLikeType0.withStaticTyping();
      assertNotSame(collectionLikeType1, collectionLikeType0);
      assertTrue(collectionLikeType1.useStaticType());
      assertTrue(collectionLikeType1.equals((Object)collectionLikeType0));
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(collectionType0, collectionType0);
      CollectionLikeType collectionLikeType1 = collectionLikeType0.withStaticTyping();
      CollectionLikeType collectionLikeType2 = collectionLikeType1.withStaticTyping();
      assertTrue(collectionLikeType2.equals((Object)collectionLikeType0));
      assertTrue(collectionLikeType2.useStaticType());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Integer integer0 = new Integer((-4873));
      CollectionType collectionType1 = collectionType0.withContentValueHandler(integer0);
      boolean boolean0 = collectionType1.hasHandlers();
      assertFalse(collectionType0.hasHandlers());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      String string0 = collectionType0.buildCanonicalName();
      assertFalse(collectionType0.hasHandlers());
      assertEquals("java.util.LinkedList<java.lang.Object>", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      boolean boolean0 = collectionType0.equals((Object) null);
      assertFalse(collectionType0.hasHandlers());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      CollectionType collectionType1 = typeFactory0.constructCollectionType((Class<? extends Collection>) class0, (JavaType) collectionType0);
      boolean boolean0 = collectionType0.equals(collectionType1);
      assertFalse(collectionType1.hasHandlers());
      assertFalse(boolean0);
  }
}
