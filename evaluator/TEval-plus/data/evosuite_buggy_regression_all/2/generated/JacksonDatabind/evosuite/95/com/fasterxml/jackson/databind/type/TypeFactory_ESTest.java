/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:05:14 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.cfg.ContextAttributes;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.ClassStack;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.PlaceholderForType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.type.TypeModifier;
import com.fasterxml.jackson.databind.util.LRUMap;
import java.lang.reflect.Array;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Properties;
import java.util.TreeSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeFactory_ESTest extends TypeFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<AnnotationIntrospector.ReferenceProperty.Type> class0 = AnnotationIntrospector.ReferenceProperty.Type.class;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class0, class0);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<RuntimeException> class0 = RuntimeException.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      JavaType javaType0 = typeFactory0.constructSimpleType(class0, class0, javaTypeArray0);
      assertFalse(javaType0.isFinal());
      assertFalse(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<LinkedList> class0 = LinkedList.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      assertFalse(collectionType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<HashSet> class0 = HashSet.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.reader();
      ObjectReader objectReader1 = objectReader0.forType(class0);
      assertFalse(objectReader1.equals((Object)objectReader0));
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<MapLikeType> class0 = MapLikeType.class;
      Class<ObjectMapper.DefaultTyping>[] classArray0 = (Class<ObjectMapper.DefaultTyping>[]) Array.newInstance(Class.class, 1);
      Class<ObjectMapper.DefaultTyping> class1 = ObjectMapper.DefaultTyping.class;
      classArray0[0] = class1;
      // Undeclared exception!
      try { 
        typeFactory0.constructParametricType(class0, classArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class com.fasterxml.jackson.databind.type.MapLikeType with 1 type parameter: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ReferenceType> class0 = ReferenceType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      assertFalse(mapLikeType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<RuntimeException> class0 = RuntimeException.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      JavaType javaType0 = typeFactory0.constructParametrizedType(class0, class0, javaTypeArray0);
      assertFalse(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ReferenceType> class0 = ReferenceType.class;
      Class<MapLikeType> class1 = MapLikeType.class;
      Class<MapType>[] classArray0 = (Class<MapType>[]) Array.newInstance(Class.class, 15);
      // Undeclared exception!
      try { 
        typeFactory0.constructParametrizedType(class0, class1, classArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<HashSet> class0 = HashSet.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      byte[] byteArray0 = objectMapper0.writeValueAsBytes(class0);
      assertEquals(19, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      typeFactory0.clearCache();
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<AnnotationIntrospector.ReferenceProperty.Type> class0 = AnnotationIntrospector.ReferenceProperty.Type.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      JavaType javaType0 = typeFactory0.moreSpecificType(collectionLikeType0, (JavaType) null);
      assertNotNull(javaType0);
      assertTrue(javaType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0._findPrimitive("boolean");
      assertNotNull(class0);
      
      Class<Object> class1 = Object.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructCollectionLikeType(class0, class1);
      assertEquals("boolean", class0.toString());
      assertTrue(collectionLikeType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<SerializationFeature> class0 = SerializationFeature.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      assertTrue(arrayType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      // Undeclared exception!
      try { 
        typeFactory0.constructType((TypeReference<?>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructRawMapType(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      try { 
        typeFactory0.findClass("Attempt to write plain `java.math.BigDecimal` (see JsonGenerator.Feature.WRITE_BIGDECIMAL_AS_PLAIN) with illegal scale (%d): needs to be between [-%d, %d]");
        fail("Expecting exception: ClassNotFoundException");
      
      } catch(ClassNotFoundException e) {
         //
         // Attempt to write plain `java/math/BigDecimal` (see JsonGenerator/Feature/WRITE_BIGDECIMAL_AS_PLAIN) with illegal scale (%d): needs to be between [-%d, %d]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      assertFalse(mapLikeType0.hasValueHandler());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.constructFromCanonical("long");
      assertTrue(javaType0.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassLoader classLoader0 = ClassLoader.getSystemClassLoader();
      TypeFactory typeFactory1 = typeFactory0.withClassLoader(classLoader0);
      try { 
        typeFactory1.findClass("Illegal unquoted character (");
        fail("Expecting exception: ClassNotFoundException");
      
      } catch(ClassNotFoundException e) {
         //
         // Illegal unquoted character (
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<RuntimeException> class0 = RuntimeException.class;
      Class<SerializationFeature> class1 = SerializationFeature.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, (JavaType[]) null);
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class0, class1, typeBindings0);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructReferenceType(class0, (JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.ReferenceType", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeFactory typeFactory1 = typeFactory0.withCache((LRUMap<Object, JavaType>) null);
      assertFalse(typeFactory1.equals((Object)typeFactory0));
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashSet> class0 = HashSet.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0);
      ArrayType arrayType0 = typeFactory0.constructArrayType(javaType0);
      assertFalse(arrayType0.isCollectionLikeType());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      LRUMap<Object, JavaType> lRUMap0 = new LRUMap<Object, JavaType>(2000, 2000);
      TypeFactory typeFactory0 = new TypeFactory(lRUMap0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TypeFactory typeFactory0 = new TypeFactory((LRUMap<Object, JavaType>) null);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      TypeFactory typeFactory1 = typeFactory0.withModifier((TypeModifier) null);
      assertNotSame(typeFactory0, typeFactory1);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      TypeModifier typeModifier1 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory2 = typeFactory1.withModifier(typeModifier1);
      assertFalse(typeFactory2.equals((Object)typeFactory0));
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      Class<?> class0 = TypeFactory.rawClass(simpleType0);
      assertFalse(class0.isInterface());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<Short> class0 = Short.TYPE;
      Class<?> class1 = TypeFactory.rawClass(class0);
      assertFalse(class1.isAnnotation());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      try { 
        typeFactory0.findClass("");
        fail("Expecting exception: ClassNotFoundException");
      
      } catch(ClassNotFoundException e) {
         //
         // 
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("int");
      assertFalse(class0.isInterface());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("float");
      assertEquals("float", class0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("double");
      assertEquals("double", class0.toString());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<?> class0 = typeFactory0.findClass("byte");
      assertEquals("byte", class0.toString());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("char");
      assertEquals("char", class0.toString());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      typeFactory0._findPrimitive("short");
      JavaType javaType0 = typeFactory0.constructFromCanonical("short");
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<?> class0 = typeFactory0.findClass("void");
      assertEquals("void", class0.toString());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0);
      JavaType javaType1 = typeFactory0.constructSpecializedType(javaType0, class0);
      assertFalse(javaType1.isArrayType());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_OBJECT;
      Class<RuntimeException> class0 = RuntimeException.class;
      JavaType javaType0 = typeFactory0.constructSpecializedType(simpleType0, class0);
      assertFalse(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      Class<String> class1 = String.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class1);
      // Undeclared exception!
      try { 
        typeFactory0.constructSpecializedType(javaType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class java.util.Properties not subtype of [simple type, class java.lang.String]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      Class<Integer> class0 = Integer.class;
      JavaType javaType0 = typeFactory0.constructSpecializedType(simpleType0, class0);
      assertFalse(javaType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<TreeSet> class0 = TreeSet.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      Class<Object> class1 = Object.class;
      typeFactory0.constructGeneralizedType(collectionType0, class1);
      assertEquals(1, collectionType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      PlaceholderForType placeholderForType0 = new PlaceholderForType((-1972223768));
      Class<Object> class0 = Object.class;
      JavaType javaType0 = typeFactory0.constructGeneralizedType(placeholderForType0, class0);
      assertFalse(javaType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Class<AnnotationIntrospector.ReferenceProperty.Type> class0 = AnnotationIntrospector.ReferenceProperty.Type.class;
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType[] javaTypeArray0 = new JavaType[2];
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_COMPARABLE;
      ReferenceType referenceType0 = new ReferenceType(simpleType0, (JavaType) null);
      MapType mapType0 = MapType.construct((Class<?>) class0, typeBindings0, (JavaType) null, javaTypeArray0, (JavaType) referenceType0, (JavaType) simpleType0);
      Class<Object> class1 = Object.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructGeneralizedType(mapType0, class1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.ReferenceType", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      PlaceholderForType placeholderForType0 = new PlaceholderForType(3087);
      JavaType javaType0 = placeholderForType0.containedTypeOrUnknown(3087);
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructGeneralizedType(javaType0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.ObjectMapper$DefaultTyping not a super-type of [simple type, class java.lang.Object]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      JavaType javaType0 = typeFactory0.moreSpecificType((JavaType) null, (JavaType) null);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      JavaType javaType0 = typeFactory0.moreSpecificType(simpleType0, simpleType0);
      assertFalse(javaType0.isContainerType());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<AnnotationIntrospector.ReferenceProperty.Type> class0 = AnnotationIntrospector.ReferenceProperty.Type.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      JavaType javaType0 = TypeFactory.unknownType();
      typeFactory0.moreSpecificType(javaType0, collectionLikeType0);
      assertTrue(javaType0.isJavaLangObject());
      assertFalse(collectionLikeType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<AnnotationIntrospector.ReferenceProperty.Type> class0 = AnnotationIntrospector.ReferenceProperty.Type.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      JavaType javaType0 = TypeFactory.unknownType();
      JavaType javaType1 = typeFactory0.moreSpecificType(collectionLikeType0, javaType0);
      assertFalse(javaType1.isJavaLangObject());
      assertTrue(javaType0.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<Properties> class0 = Properties.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_OBJECT;
      JavaType javaType0 = typeFactory0.constructType((Type) simpleType0, (Class<?>) class0);
      assertSame(javaType0, simpleType0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      // Undeclared exception!
      try { 
        typeFactory0.constructType((Type) null, (Class<?>) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unrecognized Type: [null]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<AnnotationIntrospector.ReferenceProperty.Type> class0 = AnnotationIntrospector.ReferenceProperty.Type.class;
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) class0);
      assertFalse(javaType0.isFinal());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      JavaType javaType0 = typeFactory0.uncheckedSimpleType(class0);
      Class<DeserializationFeature> class1 = DeserializationFeature.class;
      JavaType javaType1 = typeFactory0.constructType((Type) javaType0, (Class<?>) class1);
      assertFalse(javaType1.isArrayType());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayList> class0 = ArrayList.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      assertTrue(collectionLikeType0.hasContentType());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructMapType((Class<? extends Map>) null, class0, (Class<?>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      // Undeclared exception!
      try { 
        typeFactory0.constructMapType(class0, class0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.Properties with 2 type parameters: class expects 0
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<AnnotationIntrospector.ReferenceProperty.Type> class0 = AnnotationIntrospector.ReferenceProperty.Type.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      Class<LinkedList> class1 = LinkedList.class;
      Class<Properties> class2 = Properties.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class1, class2);
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class1, (JavaType) collectionLikeType0);
      JavaType[] javaTypeArray0 = new JavaType[6];
      JavaType javaType0 = typeFactory0._constructSimple(class1, typeBindings0, collectionType0, javaTypeArray0);
      assertEquals(1, javaType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashSet> class0 = HashSet.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn(simpleType0, (JavaType) null).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      doReturn((String) null).when(typeModifier0).toString();
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      // Undeclared exception!
      try { 
        typeFactory1.constructType((Type) class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // TypeModifier null (of type com.fasterxml.jackson.databind.type.TypeModifier$MockitoMock$1144188777) return null for type [simple type, class int]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Properties> class0 = Properties.class;
      TypeModifier typeModifier0 = mock(TypeModifier.class, new ViolatedAssumptionAnswer());
      doReturn((JavaType) null).when(typeModifier0).modifyType(any(com.fasterxml.jackson.databind.JavaType.class) , any(java.lang.reflect.Type.class) , any(com.fasterxml.jackson.databind.type.TypeBindings.class) , any(com.fasterxml.jackson.databind.type.TypeFactory.class));
      doReturn((String) null).when(typeModifier0).toString();
      TypeFactory typeFactory1 = typeFactory0.withModifier(typeModifier0);
      // Undeclared exception!
      try { 
        typeFactory1.constructType((Type) class0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // TypeModifier null (of type com.fasterxml.jackson.databind.type.TypeModifier$MockitoMock$1144188777) return null for type [simple type, class java.lang.Object]
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LongNode> class0 = LongNode.class;
      Class<MapType> class1 = MapType.class;
      JavaType[] javaTypeArray0 = typeFactory0.findTypeParameters(class0, class1);
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ContextAttributes contextAttributes0 = ContextAttributes.getEmpty();
      ObjectReader objectReader0 = objectMapper0.reader(contextAttributes0);
      Class<RuntimeException> class0 = RuntimeException.class;
      objectReader0.forType(class0);
      JavaType javaType0 = typeFactory0.constructType((Type) class0);
      assertTrue(javaType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedList> class0 = LinkedList.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ArrayType arrayType0 = new ArrayType(simpleType0, typeBindings0, typeBindings0, simpleType0, class0, false);
      CollectionType collectionType0 = new CollectionType(arrayType0, simpleType0);
      CollectionType collectionType1 = typeFactory0.constructCollectionType((Class<? extends Collection>) class0, (JavaType) collectionType0);
      assertTrue(collectionType1.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<MapLikeType> class0 = MapLikeType.class;
      ClassStack classStack0 = new ClassStack(class0);
      ParameterizedType parameterizedType0 = mock(ParameterizedType.class, new ViolatedAssumptionAnswer());
      doReturn((Type[]) null).when(parameterizedType0).getActualTypeArguments();
      doReturn(class0).when(parameterizedType0).getRawType();
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      JavaType javaType0 = typeFactory0._fromParamType(classStack0, parameterizedType0, typeBindings0);
      assertFalse(javaType0.isFinal());
  }
}
