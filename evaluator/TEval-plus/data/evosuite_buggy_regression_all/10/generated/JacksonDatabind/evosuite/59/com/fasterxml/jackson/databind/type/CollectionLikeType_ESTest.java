/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:35:11 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.node.BigIntegerNode;
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
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CollectionLikeType_ESTest extends CollectionLikeType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      String string0 = collectionType0.getErasedSignature();
      assertEquals("Ljava/util/LinkedList;", string0);
      assertFalse(collectionType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class0, (JavaType) collectionType0);
      String string0 = collectionLikeType0.getTypeName();
      assertEquals("[collection-like type; class java.util.LinkedList, contains [collection type; class java.util.LinkedList, contains [collection type; class java.util.LinkedList, contains [simple type, class java.lang.Object]]]]", string0);
      assertFalse(collectionLikeType0.hasHandlers());
      assertEquals(1, collectionLikeType0.containedTypeCount());
      assertFalse(collectionLikeType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<SimpleType> class1 = SimpleType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class1, class0);
      CollectionLikeType collectionLikeType1 = collectionLikeType0.withContentTypeHandler(class0);
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
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Class<ArrayType> class1 = ArrayType.class;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class1, (JavaType) collectionType0);
      JavaType[] javaTypeArray0 = new JavaType[1];
      JavaType javaType0 = collectionLikeType0.refine(class0, (TypeBindings) null, collectionType0, javaTypeArray0);
      assertFalse(javaType0.hasHandlers());
      assertFalse(javaType0.useStaticType());
      assertFalse(javaType0.equals((Object)collectionLikeType0));
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      String string0 = collectionType0.getGenericSignature();
      assertEquals("Ljava/util/LinkedList<Ljava/util/LinkedList<Ljava/lang/Object;>;>;", string0);
      assertFalse(collectionType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<ReferenceType> class0 = ReferenceType.class;
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Integer> class1 = Integer.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class1, class1);
      CollectionLikeType collectionLikeType1 = collectionLikeType0.withTypeHandler(class0);
      assertTrue(collectionLikeType1.equals((Object)collectionLikeType0));
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType((Class<?>) class0, (JavaType) simpleType0);
      CollectionLikeType collectionLikeType1 = collectionLikeType0.withValueHandler(simpleType0);
      assertFalse(collectionLikeType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      Class<BigIntegerNode> class1 = BigIntegerNode.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class1, javaTypeArray0);
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
  public void test11()  throws Throwable  {
      Class<Integer> class0 = Integer.TYPE;
      JavaType javaType0 = TypeFactory.unknownType();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct(class0, javaType0);
      CollectionLikeType collectionLikeType1 = collectionLikeType0.withContentValueHandler(javaType0);
      assertFalse(collectionLikeType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Class<Integer> class1 = Integer.class;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class1, (JavaType) collectionType0);
      JavaType javaType0 = collectionLikeType0._narrow(class1);
      assertFalse(javaType0.hasHandlers());
      assertFalse(javaType0.useStaticType());
      assertTrue(javaType0.equals((Object)collectionLikeType0));
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
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
  public void test14()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(javaType0, javaType0);
      // Undeclared exception!
      try { 
        collectionLikeType0.withContentType((JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.CollectionLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class0, (JavaType) collectionType0);
      JavaType javaType0 = collectionLikeType0.withContentType(collectionType0);
      assertFalse(javaType0.useStaticType());
      assertEquals(1, javaType0.containedTypeCount());
      assertFalse(javaType0.hasHandlers());
      assertSame(javaType0, collectionLikeType0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(javaType0, javaType0);
      CollectionLikeType collectionLikeType1 = collectionLikeType0.withStaticTyping();
      CollectionLikeType collectionLikeType2 = collectionLikeType1.withStaticTyping();
      assertTrue(collectionLikeType2.useStaticType());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      CollectionType collectionType1 = collectionType0.withTypeHandler(class0);
      CollectionType collectionType2 = typeFactory0.constructCollectionType((Class<? extends Collection>) class0, (JavaType) collectionType1);
      assertFalse(collectionType0.hasHandlers());
      assertTrue(collectionType1.equals((Object)collectionType0));
      assertTrue(collectionType2.hasHandlers());
      assertFalse(collectionType2.equals((Object)collectionType1));
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapType> class0 = MapType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, (JavaType) null);
      String string0 = collectionLikeType0.buildCanonicalName();
      assertEquals("com.fasterxml.jackson.databind.type.MapType", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      String string0 = collectionType0.buildCanonicalName();
      assertFalse(collectionType0.hasHandlers());
      assertEquals("java.util.LinkedList<java.lang.Object>", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      boolean boolean0 = collectionType0.equals((Object) null);
      assertFalse(collectionType0.hasHandlers());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      CollectionLikeType collectionLikeType0 = collectionType0.withStaticTyping();
      boolean boolean0 = collectionType0.equals(collectionLikeType0);
      assertTrue(boolean0);
      assertFalse(collectionLikeType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Map> class0 = Map.class;
      Class<LinkedList> class1 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class1);
      CollectionType collectionType1 = typeFactory0.constructCollectionType(class1, class0);
      boolean boolean0 = collectionType0.equals(collectionType1);
      assertFalse(boolean0);
      assertFalse(collectionType1.hasHandlers());
  }
}
