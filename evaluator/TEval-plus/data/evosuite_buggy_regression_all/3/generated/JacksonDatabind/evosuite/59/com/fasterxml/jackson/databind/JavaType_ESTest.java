/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:41:09 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.node.IntNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.InputStream;
import java.math.RoundingMode;
import java.sql.SQLWarning;
import java.util.Collection;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JavaType_ESTest extends JavaType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<Throwable> class0 = Throwable.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      objectMapper0.readerFor((JavaType) resolvedRecursiveType0);
      assertFalse(resolvedRecursiveType0.useStaticType());
      assertTrue(resolvedRecursiveType0.isConcrete());
      assertFalse(resolvedRecursiveType0.isMapLikeType());
      assertFalse(resolvedRecursiveType0.isCollectionLikeType());
      assertFalse(resolvedRecursiveType0.hasHandlers());
      assertTrue(resolvedRecursiveType0.hasContentType());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Object object0 = new Object();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(object0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      Class<Integer> class1 = Integer.class;
      JavaType[] javaTypeArray0 = new JavaType[3];
      javaTypeArray0[2] = (JavaType) resolvedRecursiveType0;
      MapType mapType0 = MapType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, javaTypeArray0[2], (JavaType) resolvedRecursiveType0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class1, typeBindings0, (JavaType) mapType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      collectionLikeType0.getKeyType();
      assertFalse(collectionLikeType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      boolean boolean0 = javaType0.useStaticType();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      boolean boolean0 = javaType0.isFinal();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType[] javaTypeArray0 = new JavaType[8];
      javaTypeArray0[1] = (JavaType) resolvedRecursiveType0;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, javaTypeArray0[1]);
      referenceType0.getContentTypeHandler();
      assertFalse(referenceType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      resolvedRecursiveType0.getContentValueHandler();
      assertFalse(resolvedRecursiveType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Collection> class0 = Collection.class;
      Class<Object> class1 = Object.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      String string0 = collectionType0.getGenericSignature();
      assertEquals("Ljava/util/Collection<Ljava/lang/Object;>;", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<IntNode> class0 = IntNode.class;
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
  public void test09()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper((JsonFactory) null);
      DeserializationFeature deserializationFeature0 = DeserializationFeature.ACCEPT_EMPTY_ARRAY_AS_NULL_OBJECT;
      DeserializationFeature[] deserializationFeatureArray0 = new DeserializationFeature[2];
      deserializationFeatureArray0[0] = deserializationFeature0;
      deserializationFeatureArray0[1] = deserializationFeature0;
      ObjectReader objectReader0 = objectMapper0.reader(deserializationFeature0, deserializationFeatureArray0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      Class<RoundingMode> class0 = RoundingMode.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      mapLikeType0.getParameterSource();
      assertFalse(mapLikeType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      Class<Integer> class1 = Integer.class;
      JavaType[] javaTypeArray0 = new JavaType[3];
      javaTypeArray0[2] = (JavaType) resolvedRecursiveType0;
      MapType mapType0 = MapType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, javaTypeArray0[2], (JavaType) resolvedRecursiveType0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.construct((Class<?>) class1, typeBindings0, (JavaType) mapType0, javaTypeArray0, (JavaType) resolvedRecursiveType0);
      JavaType javaType0 = collectionLikeType0.forcedNarrowBy(class1);
      assertFalse(javaType0.useStaticType());
      assertSame(javaType0, collectionLikeType0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassLoader classLoader0 = ClassLoader.getSystemClassLoader();
      Class<Collection> class0 = Collection.class;
      Class<Object> class1 = Object.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      CollectionLikeType collectionLikeType0 = collectionType0.withValueHandler(classLoader0);
      Class<Integer> class2 = Integer.class;
      JavaType javaType0 = collectionLikeType0.forcedNarrowBy(class2);
      assertFalse(javaType0.useStaticType());
      assertTrue(javaType0.hasHandlers());
      assertTrue(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Collection> class0 = Collection.class;
      Class<Object> class1 = Object.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class1);
      CollectionType collectionType1 = collectionType0.withTypeHandler(typeBindings0);
      Class<Integer> class2 = Integer.class;
      JavaType javaType0 = collectionType1.forcedNarrowBy(class2);
      assertFalse(javaType0.useStaticType());
      assertTrue(javaType0.hasHandlers());
      assertFalse(javaType0.hasValueHandler());
      assertTrue(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      boolean boolean0 = resolvedRecursiveType0.isTypeOrSubTypeOf(class0);
      assertTrue(boolean0);
      assertFalse(resolvedRecursiveType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<RoundingMode> class0 = RoundingMode.class;
      boolean boolean0 = javaType0.isTypeOrSubTypeOf(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JavaType javaType0 = TypeFactory.unknownType();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(javaType0, javaType0);
      JavaType[] javaTypeArray0 = new JavaType[9];
      javaTypeArray0[0] = javaType0;
      javaTypeArray0[3] = javaType0;
      javaTypeArray0[5] = javaType0;
      MapType mapType0 = MapType.construct((Class<?>) class0, (TypeBindings) null, (JavaType) collectionLikeType0, javaTypeArray0, javaTypeArray0[3], javaTypeArray0[0]);
      Class<Throwable> class1 = Throwable.class;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class1, (TypeBindings) null, (JavaType) mapType0, javaTypeArray0, javaTypeArray0[5]);
      boolean boolean0 = referenceType0.isTypeOrSubTypeOf(class0);
      assertTrue(boolean0);
      assertTrue(collectionLikeType0.isJavaLangObject());
      assertFalse(referenceType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<InputStream> class0 = InputStream.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<Throwable> class0 = Throwable.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      boolean boolean0 = resolvedRecursiveType0.isJavaLangObject();
      assertFalse(resolvedRecursiveType0.useStaticType());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      boolean boolean0 = resolvedRecursiveType0.isJavaLangObject();
      assertTrue(boolean0);
      assertFalse(resolvedRecursiveType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JavaType javaType0 = TypeFactory.unknownType();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(javaType0, javaType0);
      JavaType[] javaTypeArray0 = new JavaType[15];
      javaTypeArray0[0] = javaType0;
      javaTypeArray0[3] = javaType0;
      MapType mapType0 = MapType.construct((Class<?>) class0, (TypeBindings) null, (JavaType) collectionLikeType0, javaTypeArray0, javaTypeArray0[3], javaTypeArray0[0]);
      boolean boolean0 = mapType0.hasGenericTypes();
      assertFalse(boolean0);
      assertTrue(collectionLikeType0.isJavaLangObject());
      assertFalse(mapType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      Class<Integer> class1 = Integer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class1);
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType((Class<?>) class0, (JavaType) simpleType0, (JavaType) simpleType0);
      boolean boolean0 = mapLikeType0.hasGenericTypes();
      assertFalse(mapLikeType0.hasHandlers());
      assertTrue(boolean0);
      assertFalse(simpleType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(javaType0, javaType0);
      JavaType javaType1 = collectionLikeType0.containedTypeOrUnknown(1536);
      assertNotNull(javaType1);
      assertTrue(collectionLikeType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      boolean boolean0 = javaType0.hasValueHandler();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      SQLWarning sQLWarning0 = new SQLWarning("");
      MapType mapType1 = mapType0.withValueHandler(sQLWarning0);
      boolean boolean0 = mapType1.hasValueHandler();
      assertFalse(mapType1.useStaticType());
      assertTrue(mapType1.hasHandlers());
      assertTrue(boolean0);
      assertFalse(mapType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<Throwable> class0 = Throwable.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ArrayType arrayType0 = ArrayType.construct((JavaType) resolvedRecursiveType0, typeBindings0);
      ArrayType arrayType1 = arrayType0.withTypeHandler(resolvedRecursiveType0);
      ArrayType arrayType2 = arrayType1.withContentTypeHandler((Object) null);
      boolean boolean0 = arrayType2.hasHandlers();
      assertFalse(arrayType0.useStaticType());
      assertFalse(arrayType0.hasHandlers());
      assertTrue(boolean0);
      assertFalse(arrayType2.hasValueHandler());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(javaType0, javaType0);
      boolean boolean0 = collectionLikeType0.hasHandlers();
      assertFalse(boolean0);
      assertTrue(collectionLikeType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ArrayType arrayType0 = ArrayType.construct((JavaType) resolvedRecursiveType0, typeBindings0);
      ArrayType arrayType1 = arrayType0.withValueHandler(class0);
      boolean boolean0 = arrayType1.hasHandlers();
      assertFalse(arrayType0.useStaticType());
      assertFalse(arrayType0.hasHandlers());
      assertTrue(boolean0);
  }
}