/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:44:03 GMT 2023
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
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.LRUMap;
import java.lang.reflect.Type;
import java.time.temporal.ChronoField;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MapLikeType_ESTest extends MapLikeType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<MapType> class0 = MapType.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, javaType0, javaType0);
      String string0 = mapLikeType0.getGenericSignature();
      assertEquals("Lcom/fasterxml/jackson/databind/type/MapType<Ljava/lang/Object;Ljava/lang/Object;>;", string0);
      assertFalse(mapLikeType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Integer> class0 = Integer.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      MapLikeType mapLikeType1 = mapLikeType0.withContentTypeHandler(typeFactory0);
      assertTrue(mapLikeType1.equals((Object)mapLikeType0));
      assertFalse(mapLikeType1.isJavaLangObject());
      assertTrue(mapLikeType1.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<MapType> class0 = MapType.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, javaType0, javaType0);
      MapLikeType mapLikeType1 = mapLikeType0.withKeyValueHandler(class0);
      assertFalse(mapLikeType1.useStaticType());
      assertTrue(mapLikeType1.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      mapType0.getContentTypeHandler();
      assertFalse(mapType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<SimpleType> class0 = SimpleType.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, javaType0, javaType0);
      JavaType javaType1 = mapLikeType0._narrow(class0);
      assertFalse(javaType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_OBJECT;
      MapType mapType0 = new MapType(simpleType0, simpleType0, simpleType0);
      String string0 = mapType0.getErasedSignature();
      assertEquals("Ljava/lang/Object;", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<String> class0 = String.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      JavaType[] javaTypeArray0 = new JavaType[2];
      javaTypeArray0[0] = (JavaType) resolvedRecursiveType0;
      MapType mapType0 = MapType.construct((Class<?>) class0, (TypeBindings) null, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0, javaTypeArray0[0]);
      Object object0 = mapType0.getContentValueHandler();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<CollectionType> class0 = CollectionType.class;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      Class<ArrayType> class1 = ArrayType.class;
      LinkedList<JavaType> linkedList0 = new LinkedList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class1, (List<JavaType>) linkedList0);
      JavaType[] javaTypeArray0 = new JavaType[2];
      mapLikeType0.refine(class1, typeBindings0, mapLikeType0, javaTypeArray0);
      assertFalse(mapLikeType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      Class<MapType> class1 = MapType.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      JavaType[] javaTypeArray0 = new JavaType[4];
      javaTypeArray0[0] = (JavaType) simpleType0;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class1, typeBindings0, (JavaType) simpleType0, javaTypeArray0, javaTypeArray0[0]);
      MapLikeType mapLikeType0 = new MapLikeType(collectionLikeType0, collectionLikeType0, simpleType0);
      MapLikeType mapLikeType1 = mapLikeType0.withContentValueHandler(class0);
      assertTrue(mapLikeType1.hasHandlers());
      assertTrue(mapLikeType1.equals((Object)mapLikeType0));
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(simpleType0, simpleType0, simpleType0);
      String string0 = mapLikeType0.toString();
      assertEquals("[map-like type; class java.lang.Enum, [simple type, class java.lang.Enum] -> [simple type, class java.lang.Enum]]", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class0);
      CollectionType collectionType0 = new CollectionType(collectionLikeType0, collectionLikeType0);
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(collectionLikeType0, collectionLikeType0, collectionType0);
      MapLikeType mapLikeType1 = MapLikeType.construct(class0, mapLikeType0, mapLikeType0);
      MapLikeType mapLikeType2 = mapLikeType1.withKeyTypeHandler(typeFactory0);
      assertTrue(mapLikeType2.hasHandlers());
      assertFalse(mapLikeType2.useStaticType());
      assertTrue(mapLikeType2.equals((Object)mapLikeType1));
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      // Undeclared exception!
      try { 
        MapLikeType.upgradeFrom((JavaType) null, (JavaType) null, (JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.MapLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0);
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, javaType0, javaType0);
      assertFalse(mapLikeType0.useStaticType());
      assertEquals(2, mapLikeType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<CollectionType> class0 = CollectionType.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, javaType0, javaType0);
      Class<String> class1 = String.class;
      SimpleType simpleType0 = new SimpleType(class1);
      MapLikeType mapLikeType1 = mapLikeType0.withKeyType(simpleType0);
      assertFalse(mapLikeType1.equals((Object)mapLikeType0));
      assertNotSame(mapLikeType1, mapLikeType0);
      assertFalse(mapLikeType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(simpleType0, simpleType0, simpleType0);
      MapLikeType mapLikeType1 = mapLikeType0.withKeyType(simpleType0);
      assertSame(mapLikeType1, mapLikeType0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      LRUMap<Object, JavaType> lRUMap0 = new LRUMap<Object, JavaType>(761, 761);
      TypeFactory typeFactory0 = new TypeFactory(lRUMap0);
      Class<ArrayType> class0 = ArrayType.class;
      Class<ResolvedRecursiveType> class1 = ResolvedRecursiveType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class1);
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(collectionLikeType0, collectionLikeType0, collectionLikeType0);
      JavaType javaType0 = mapLikeType0.withContentType(mapLikeType0);
      assertFalse(javaType0.equals((Object)mapLikeType0));
      assertNotSame(javaType0, mapLikeType0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<MapLikeType> class0 = MapLikeType.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, javaType0, javaType0);
      JavaType javaType1 = mapLikeType0.withContentType(javaType0);
      assertSame(javaType1, mapLikeType0);
      assertFalse(javaType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      LRUMap<Object, JavaType> lRUMap0 = new LRUMap<Object, JavaType>(761, 761);
      TypeFactory typeFactory0 = new TypeFactory(lRUMap0);
      Class<ArrayType> class0 = ArrayType.class;
      Class<ResolvedRecursiveType> class1 = ResolvedRecursiveType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class1);
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(collectionLikeType0, collectionLikeType0, collectionLikeType0);
      MapLikeType mapLikeType1 = mapLikeType0.withStaticTyping();
      assertTrue(mapLikeType1.useStaticType());
      assertTrue(mapLikeType1.equals((Object)mapLikeType0));
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      Class<Object> class0 = Object.class;
      Class<ArrayType> class1 = ArrayType.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class1, (JavaType) simpleType0);
      MapLikeType mapLikeType0 = new MapLikeType(class0, typeBindings0, simpleType0, (JavaType[]) null, simpleType0, simpleType0, class1, class1, true);
      MapType mapType0 = new MapType(simpleType0, simpleType0, mapLikeType0);
      MapLikeType mapLikeType1 = mapType0.withStaticTyping();
      assertTrue(mapLikeType1.hasHandlers());
      assertTrue(mapLikeType1.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class0);
      CollectionType collectionType0 = new CollectionType(collectionLikeType0, collectionLikeType0);
      MapType mapType0 = new MapType(collectionType0, (JavaType) null, (JavaType) null);
      String string0 = mapType0.buildCanonicalName();
      assertEquals("com.fasterxml.jackson.databind.type.ResolvedRecursiveType", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class0);
      CollectionType collectionType0 = new CollectionType(collectionLikeType0, collectionLikeType0);
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(collectionLikeType0, collectionLikeType0, collectionType0);
      String string0 = mapLikeType0.buildCanonicalName();
      assertEquals("com.fasterxml.jackson.databind.type.ResolvedRecursiveType<com.fasterxml.jackson.databind.type.ResolvedRecursiveType<com.fasterxml.jackson.databind.type.ResolvedRecursiveType>,com.fasterxml.jackson.databind.type.ResolvedRecursiveType<com.fasterxml.jackson.databind.type.ResolvedRecursiveType<com.fasterxml.jackson.databind.type.ResolvedRecursiveType>>>", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      LRUMap<Object, JavaType> lRUMap0 = new LRUMap<Object, JavaType>(761, 761);
      TypeFactory typeFactory0 = new TypeFactory(lRUMap0);
      Class<ArrayType> class0 = ArrayType.class;
      Class<ResolvedRecursiveType> class1 = ResolvedRecursiveType.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class1);
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(collectionLikeType0, collectionLikeType0, collectionLikeType0);
      MapLikeType mapLikeType1 = mapLikeType0.withValueHandler(lRUMap0);
      Class<ArrayList> class2 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType((Class<? extends Collection>) class2, (JavaType) mapLikeType1);
      assertTrue(mapLikeType1.equals((Object)mapLikeType0));
      assertTrue(collectionType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = (MapType)typeFactory0.constructType((Type) class0);
      Class<Integer> class1 = Integer.class;
      Class<Integer> class2 = Integer.class;
      typeFactory0.constructMapType((Class<? extends Map>) class0, (JavaType) mapType0, (JavaType) mapType0);
      MapType mapType1 = new MapType(mapType0, mapType0, mapType0);
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      mapType1.equals(mapType0);
      Class<Integer> class3 = Integer.class;
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      Class<ChronoField> class1 = ChronoField.class;
      Class<CollectionLikeType> class2 = CollectionLikeType.class;
      MapType mapType0 = (MapType)typeFactory0.constructMapLikeType(class0, class1, class2);
      MapType mapType1 = (MapType)mapType0.withContentTypeHandler(class2);
      typeFactory0.constructMapType((Class<? extends Map>) class0, (JavaType) mapType1, (JavaType) mapType1);
      assertFalse(mapType0.hasHandlers());
      assertTrue(mapType1.equals((Object)mapType0));
      assertTrue(mapType1.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassLoader classLoader0 = ClassLoader.getSystemClassLoader();
      typeFactory0.withClassLoader(classLoader0);
      Class<Object> class0 = Object.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      ArrayType arrayType1 = arrayType0.withTypeHandler(typeFactory0);
      Class<HashMap> class1 = HashMap.class;
      MapType mapType0 = typeFactory0.constructMapType((Class<? extends Map>) class1, (JavaType) arrayType1, (JavaType) arrayType0);
      assertTrue(mapType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      Class<CollectionType> class0 = CollectionType.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      boolean boolean0 = mapLikeType0.equals((Object) null);
      assertFalse(boolean0);
      assertFalse(mapLikeType0.useStaticType());
  }
}
