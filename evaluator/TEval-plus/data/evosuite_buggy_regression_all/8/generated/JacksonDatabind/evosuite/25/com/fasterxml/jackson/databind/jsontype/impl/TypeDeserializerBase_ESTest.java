/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:05:09 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.JsonParserDelegate;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.deser.AbstractDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.ext.CoreXMLDeserializers;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.TypeNameIdResolver;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.PipedReader;
import java.util.EnumSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeDeserializerBase_ESTest extends TypeDeserializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<ClassNameIdResolver> class0 = ClassNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      CollectionType collectionType0 = CollectionType.construct(class0, simpleType0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionType0, typeFactory0);
      Class<String> class1 = String.class;
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = new AsExternalTypeDeserializer(collectionType0, minimalClassNameIdResolver0, "uea.naEJi~&A(dy7sVT", false, class1);
      String string0 = asExternalTypeDeserializer0.getPropertyName();
      assertEquals("uea.naEJi~&A(dy7sVT", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<TypeNameIdResolver> class1 = TypeNameIdResolver.class;
      JavaType javaType0 = TypeFactory.unknownType();
      JavaType javaType1 = typeFactory0.constructReferenceType(class1, javaType0);
      CollectionType collectionType0 = CollectionType.construct(class0, javaType1);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType1, typeFactory0);
      Class<CoreXMLDeserializers.Std> class2 = CoreXMLDeserializers.Std.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(collectionType0, classNameIdResolver0, "; base-type:", true, class2);
      TypeIdResolver typeIdResolver0 = asPropertyTypeDeserializer0.getTypeIdResolver();
      assertSame(classNameIdResolver0, typeIdResolver0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionType0, typeFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(collectionType0, classNameIdResolver0, "KhW2A", false, class0);
      String string0 = asPropertyTypeDeserializer0.toString();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionType0, typeFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(collectionType0, classNameIdResolver0, "5|", false, class0);
      String string0 = asPropertyTypeDeserializer0.baseTypeName();
      assertEquals("java.util.EnumSet", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      Class<AbstractDeserializer> class0 = AbstractDeserializer.class;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(javaType0, classNameIdResolver0, "^uk 54lzxlxn`l", true, class0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, (Object) null, true);
      PipedReader pipedReader0 = new PipedReader(2);
      ObjectMapper objectMapper0 = new ObjectMapper();
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, pipedReader0, objectMapper0, charsToNameCanonicalizer0);
      JsonParserDelegate jsonParserDelegate0 = new JsonParserDelegate(readerBasedJsonParser0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0._deserializeWithNativeTypeId(jsonParserDelegate0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<TypeNameIdResolver> class0 = TypeNameIdResolver.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(simpleType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, classNameIdResolver0, "", false, class0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer1 = new AsWrapperTypeDeserializer(asWrapperTypeDeserializer0, (BeanProperty) null);
      assertEquals(JsonTypeInfo.As.WRAPPER_OBJECT, asWrapperTypeDeserializer1.getTypeInclusion());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionType0, typeFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(collectionType0, classNameIdResolver0, "KhW2A", false, class0);
      Class<?> class1 = asPropertyTypeDeserializer0.getDefaultImpl();
      assertFalse(class1.isInterface());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionType0, typeFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(collectionType0, classNameIdResolver0, "", false, class0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      PipedReader pipedReader0 = new PipedReader();
      ObjectMapper objectMapper0 = new ObjectMapper();
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, pipedReader0, objectMapper0, charsToNameCanonicalizer0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._deserializeWithNativeTypeId(readerBasedJsonParser0, (DeserializationContext) null, collectionType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, (TypeFactory) null);
      Class<AbstractDeserializer> class0 = AbstractDeserializer.class;
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, (Object) null, true);
      PipedReader pipedReader0 = new PipedReader();
      ObjectMapper objectMapper0 = new ObjectMapper();
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, pipedReader0, objectMapper0, charsToNameCanonicalizer0);
      JsonParserDelegate jsonParserDelegate0 = new JsonParserDelegate(readerBasedJsonParser0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, classNameIdResolver0, "^uk 54lzxlxn`l", true, class0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._deserializeWithNativeTypeId(jsonParserDelegate0, defaultDeserializationContext_Impl0, "^uk 54lzxlxn`l");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionType0, typeFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(collectionType0, classNameIdResolver0, "5|", false, class0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._handleUnknownTypeId((DeserializationContext) null, "aB:B", (TypeIdResolver) null, collectionType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<EnumSet> class0 = EnumSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(collectionType0, typeFactory0);
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(collectionType0, classNameIdResolver0, "5|", false, class0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._handleUnknownTypeId((DeserializationContext) null, "5|", classNameIdResolver0, collectionType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }
}