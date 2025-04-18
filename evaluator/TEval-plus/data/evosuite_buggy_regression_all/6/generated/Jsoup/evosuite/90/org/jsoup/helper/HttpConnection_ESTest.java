/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:39:41 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.Proxy;
import java.net.URL;
import java.net.UnknownServiceException;
import java.nio.charset.IllegalCharsetNameException;
import java.util.Collection;
import java.util.ConcurrentModificationException;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import javax.net.ssl.SSLSocketFactory;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.net.MockURL;
import org.evosuite.runtime.testdata.EvoSuiteURL;
import org.evosuite.runtime.testdata.NetworkHandling;
import org.jsoup.Connection;
import org.jsoup.helper.HttpConnection;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class HttpConnection_ESTest extends HttpConnection_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.followRedirects(true);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      EvoSuiteURL evoSuiteURL0 = new EvoSuiteURL("http://www.someFakeButWellFormedURL.org/fooExample");
      NetworkHandling.createRemoteTextFile(evoSuiteURL0, "\r\n\r\n");
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection_Request0.url = uRL0;
      HttpConnection.Response httpConnection_Response0 = HttpConnection.Response.execute(httpConnection_Request0);
      httpConnection_Response0.parse();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.sslSocketFactory((SSLSocketFactory) null);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      boolean boolean0 = httpConnection_Request0.ignoreContentType();
      assertEquals(1048576, httpConnection_Request0.maxBodySize());
      assertTrue(httpConnection_Request0.followRedirects());
      assertFalse(boolean0);
      assertEquals("UTF-8", httpConnection_Request0.postDataCharset());
      assertEquals(30000, httpConnection_Request0.timeout());
      assertFalse(httpConnection_Request0.ignoreHttpErrors());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Parser parser0 = Parser.xmlParser();
      Connection connection0 = httpConnection0.parser(parser0);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.ignoreContentType(false);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      Proxy proxy0 = Proxy.NO_PROXY;
      httpConnection_Request0.proxy(proxy0);
      URL uRL0 = MockURL.getHttpExample();
      httpConnection_Request0.url = uRL0;
      try { 
        HttpConnection.Response.execute(httpConnection_Request0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // java.lang.UnsupportedOperationException: Method not implemented.
         //
         verifyException("org.evosuite.runtime.mock.java.net.MockURL", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      boolean boolean0 = httpConnection_Request0.followRedirects();
      assertEquals(30000, httpConnection_Request0.timeout());
      assertFalse(httpConnection_Request0.ignoreHttpErrors());
      assertEquals(1048576, httpConnection_Request0.maxBodySize());
      assertTrue(boolean0);
      assertEquals("UTF-8", httpConnection_Request0.postDataCharset());
      assertFalse(httpConnection_Request0.ignoreContentType());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      HttpConnection.Request httpConnection_Request1 = (HttpConnection.Request)httpConnection_Request0.removeCookie("HXi!/1@j}3,|");
      assertEquals("UTF-8", httpConnection_Request1.postDataCharset());
      assertEquals(1048576, httpConnection_Request1.maxBodySize());
      assertFalse(httpConnection_Request1.ignoreContentType());
      assertFalse(httpConnection_Request1.ignoreHttpErrors());
      assertTrue(httpConnection_Request1.followRedirects());
      assertEquals(30000, httpConnection_Request1.timeout());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      Connection connection0 = httpConnection0.response((Connection.Response) httpConnection_Response0);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.cookie("Content-Type", "multipart/form-data");
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      // Undeclared exception!
      try { 
        httpConnection0.get();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // URL must be specified to connect
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.maxBodySize(32768);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      URL uRL0 = MockURL.getFileExample();
      Connection connection0 = HttpConnection.connect(uRL0);
      assertNotNull(connection0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.proxy("FxJe>#,'d0o=>bN$", 2157);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.userAgent("{dr@");
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      // Undeclared exception!
      try { 
        HttpConnection.connect("multipart/form-data");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Malformed URL: multipart/form-data
         //
         verifyException("org.jsoup.helper.HttpConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.ignoreHttpErrors(true);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.requestBody("Content-Encoding");
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      // Undeclared exception!
      try { 
        httpConnection0.postDataCharset("Content-Encoding");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // Content-Encoding
         //
         verifyException("org.jsoup.helper.HttpConnection$Request", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection.Response connection_Response0 = httpConnection0.response();
      assertNull(connection_Response0.charset());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      // Undeclared exception!
      try { 
        httpConnection0.post();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // URL must be specified to connect
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.timeout(248);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      Connection connection0 = httpConnection0.data("FZJe>.,'dznK0j=>N$", "FZJe>.,'dznK0j=>N$", (InputStream) pipedInputStream0);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.data("multipart/form-data", "application/x-www-form-urlencoded");
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection.Method connection_Method0 = Connection.Method.PATCH;
      Connection connection0 = httpConnection0.method(connection_Method0);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      URL uRL0 = MockURL.getHttpExample();
      URL uRL1 = HttpConnection.encodeUrl(uRL0);
      assertNotSame(uRL1, uRL0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.request((Connection.Request) null);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.proxy((Proxy) null);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.referrer("#/0zxZ9+1B-<H6");
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.data("org.jsoup.parser.HtmlTreeBuilder", "\r\n\r\n", (InputStream) null, "org.jsoup.parser.HtmlTreeBuilder");
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      HttpConnection.Response httpConnection_Response1 = httpConnection_Response0.charset("Content-Type");
      assertNull(httpConnection_Response1.contentType());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      // Undeclared exception!
      try { 
        httpConnection_Response0.bufferUp();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Request must be executed (with .execute(), .get(), or .post() before getting response body
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      // Undeclared exception!
      try { 
        httpConnection_Response0.bodyStream();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Request must be executed (with .execute(), .get(), or .post() before getting response body
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      String string0 = httpConnection_Response0.statusMessage();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      // Undeclared exception!
      try { 
        httpConnection_Response0.bodyAsBytes();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Request must be executed (with .execute(), .get(), or .post() before getting response body
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      String string0 = httpConnection_Response0.charset();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      int int0 = httpConnection_Response0.statusCode();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36", "org.jsoup.nodes.FormElement");
      httpConnection_KeyVal0.inputStream();
      assertEquals("org.jsoup.nodes.FormElement", httpConnection_KeyVal0.value());
      assertEquals("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36", httpConnection_KeyVal0.key());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 0, (byte)124);
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("e{xRf", "main", (InputStream) byteArrayInputStream0);
      httpConnection_KeyVal0.contentType();
      assertEquals("main", httpConnection_KeyVal0.value());
      assertEquals("e{xRf", httpConnection_KeyVal0.key());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("ISO-8859-1", "ISO-8859-1");
      String string0 = httpConnection_KeyVal0.toString();
      assertEquals("ISO-8859-1=ISO-8859-1", string0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      hashMap0.put("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36", "multipart/form-data");
      Connection connection0 = httpConnection0.data((Map<String, String>) hashMap0);
      Connection.KeyVal connection_KeyVal0 = connection0.data("Content-Type");
      assertNull(connection_KeyVal0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      String[] stringArray0 = new String[1];
      // Undeclared exception!
      try { 
        httpConnection0.data(stringArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must supply an even number of key value pairs
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36";
      stringArray0[1] = "application/x-www-form-urlencoded";
      Connection connection0 = httpConnection0.data(stringArray0);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      LinkedHashSet<Connection.KeyVal> linkedHashSet0 = new LinkedHashSet<Connection.KeyVal>();
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("%DW)$(?V/Pbv%C?", ")(m0&9UcA!V%/.M=vr");
      linkedHashSet0.add(httpConnection_KeyVal0);
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.data((Collection<Connection.KeyVal>) linkedHashSet0);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection.KeyVal connection_KeyVal0 = httpConnection0.data("FxJep#,?d0o=>bN$");
      assertNull(connection_KeyVal0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      hashMap0.put("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36", "|onv0ruv\"!]r]?s");
      httpConnection0.data((Map<String, String>) hashMap0);
      Connection.KeyVal connection_KeyVal0 = httpConnection0.data("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36");
      assertNotNull(connection_KeyVal0);
      assertEquals("|onv0ruv\"!]r]?s", connection_KeyVal0.value());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      hashMap0.put("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36", "multipart/form-data");
      Connection connection0 = httpConnection0.headers(hashMap0);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      hashMap0.put("application/x-www-form-urlencoded", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36");
      Connection connection0 = httpConnection0.cookies(hashMap0);
      assertSame(connection0, httpConnection0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      HttpConnection httpConnection0 = new HttpConnection();
      Connection connection0 = httpConnection0.header("Content-Encoding", (String) null);
      assertSame(httpConnection0, connection0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection_Request0.url = uRL0;
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.cookie("^r,", "^r,");
      try { 
        HttpConnection.Response.execute(httpConnection_Request0, httpConnection_Response0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // HTTP error fetching URL
         //
         verifyException("org.jsoup.helper.HttpConnection$Response", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      // Undeclared exception!
      try { 
        httpConnection_Request0.timeout((-2169));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Timeout milliseconds must be 0 (infinite) or greater
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      // Undeclared exception!
      try { 
        httpConnection_Request0.maxBodySize((-2028));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // maxSize must be 0 (unlimited) or larger
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      assertEquals("UTF-8", httpConnection_Request0.postDataCharset());
      
      httpConnection_Request0.postDataCharset("ISO-8859-1");
      assertEquals(1048576, httpConnection_Request0.maxBodySize());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      URL uRL0 = MockURL.getFtpExample();
      httpConnection_Request0.url = uRL0;
      try { 
        HttpConnection.Response.execute(httpConnection_Request0);
        fail("Expecting exception: MalformedURLException");
      
      } catch(MalformedURLException e) {
         //
         // Only http & https protocols supported
         //
         verifyException("org.jsoup.helper.HttpConnection$Response", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection_Request0.url = uRL0;
      Connection.Method connection_Method0 = Connection.Method.PUT;
      HttpConnection.Request httpConnection_Request1 = (HttpConnection.Request)httpConnection_Request0.method(connection_Method0);
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("))R+sV,a", "))R+sV,a");
      HttpConnection.Request httpConnection_Request2 = httpConnection_Request1.data((Connection.KeyVal) httpConnection_KeyVal0);
      try { 
        HttpConnection.Response.execute(httpConnection_Request2);
        fail("Expecting exception: UnknownServiceException");
      
      } catch(UnknownServiceException e) {
         //
         // protocol doesn't support output
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection_Request0.url = uRL0;
      Connection.Method connection_Method0 = Connection.Method.PUT;
      HttpConnection.Request httpConnection_Request1 = (HttpConnection.Request)httpConnection_Request0.method(connection_Method0);
      httpConnection_Request1.requestBody("))RsV,a");
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("))RsV,a", "))RsV,a");
      httpConnection_Request1.data((Connection.KeyVal) httpConnection_KeyVal0);
      try { 
        HttpConnection.Response.execute(httpConnection_Request0);
        fail("Expecting exception: UnknownServiceException");
      
      } catch(UnknownServiceException e) {
         //
         // protocol doesn't support output
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection_Request0.ignoreHttpErrors(true);
      httpConnection_Request0.url = uRL0;
      try { 
        HttpConnection.Response.execute(httpConnection_Request0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Could not find: www.someFakeButWellFormedURL.org
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      EvoSuiteURL evoSuiteURL0 = new EvoSuiteURL("http://www.someFakeButWellFormedURL.org/fooExample");
      NetworkHandling.createRemoteTextFile(evoSuiteURL0, "\r\n\r\n");
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      Connection.Method connection_Method0 = Connection.Method.HEAD;
      httpConnection_Request0.method(connection_Method0);
      URL uRL0 = MockURL.getHttpExample();
      httpConnection_Request0.url = uRL0;
      HttpConnection.Response.execute(httpConnection_Request0);
      assertFalse(httpConnection_Request0.ignoreHttpErrors());
      assertEquals(1048576, httpConnection_Request0.maxBodySize());
      assertFalse(httpConnection_Request0.ignoreContentType());
      assertTrue(httpConnection_Request0.followRedirects());
      assertEquals(30000, httpConnection_Request0.timeout());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      EvoSuiteURL evoSuiteURL0 = new EvoSuiteURL("http://www.someFakeButWellFormedURL.org/fooExample");
      NetworkHandling.createRemoteTextFile(evoSuiteURL0, "\r\n\r\n");
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection_Request0.url = uRL0;
      HttpConnection.Response httpConnection_Response0 = HttpConnection.Response.execute(httpConnection_Request0);
      httpConnection_Response0.body();
      httpConnection_Response0.parse();
      httpConnection_Response0.body();
      assertEquals(30000, httpConnection_Request0.timeout());
      assertEquals("UTF-8", httpConnection_Request0.postDataCharset());
      assertFalse(httpConnection_Request0.ignoreContentType());
      assertFalse(httpConnection_Request0.ignoreHttpErrors());
      assertEquals(1048576, httpConnection_Request0.maxBodySize());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      EvoSuiteURL evoSuiteURL0 = new EvoSuiteURL("http://www.someFakeButWellFormedURL.org/fooExample");
      NetworkHandling.createRemoteTextFile(evoSuiteURL0, "\r\n\r\n");
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      URL uRL0 = MockURL.getHttpExample();
      httpConnection_Request0.url = uRL0;
      HttpConnection.Response httpConnection_Response0 = HttpConnection.Response.execute(httpConnection_Request0);
      httpConnection_Response0.body();
      String string0 = httpConnection_Response0.body();
      assertEquals(1048576, httpConnection_Request0.maxBodySize());
      assertFalse(httpConnection_Request0.ignoreContentType());
      assertEquals(30000, httpConnection_Request0.timeout());
      assertEquals("\r\n\r\n", string0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      HashMap<String, List<String>> hashMap0 = new HashMap<String, List<String>>();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      hashMap0.put((String) null, linkedList0);
      httpConnection_Response0.processResponseHeaders(hashMap0);
      assertNull(httpConnection_Response0.charset());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      HttpConnection.Response httpConnection_Response1 = (HttpConnection.Response)httpConnection_Response0.addHeader("k8<,wes{", "k8<,wes{");
      Map<String, List<String>> map0 = httpConnection_Response1.headers;
      // Undeclared exception!
      try { 
        httpConnection_Response1.processResponseHeaders(map0);
        fail("Expecting exception: ConcurrentModificationException");
      
      } catch(ConcurrentModificationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayList$Itr", e);
      }
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      HttpConnection.Response httpConnection_Response1 = (HttpConnection.Response)httpConnection_Response0.addHeader("set-cookie", "=xo)e%v_`ld+5w");
      Map<String, List<String>> map0 = httpConnection_Response1.headers;
      // Undeclared exception!
      try { 
        httpConnection_Response1.processResponseHeaders(map0);
        fail("Expecting exception: ConcurrentModificationException");
      
      } catch(ConcurrentModificationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayList$Itr", e);
      }
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      HttpConnection.Response httpConnection_Response1 = (HttpConnection.Response)httpConnection_Response0.addHeader("set-cookie", "set-cookie");
      Map<String, List<String>> map0 = httpConnection_Response1.headers;
      HttpConnection.Response httpConnection_Response2 = new HttpConnection.Response();
      httpConnection_Response2.processResponseHeaders(map0);
      assertNull(httpConnection_Response2.contentType());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      URL uRL0 = MockURL.getHttpExample();
      Connection.Method connection_Method0 = Connection.Method.PATCH;
      HttpConnection.Request httpConnection_Request1 = (HttpConnection.Request)httpConnection_Request0.method(connection_Method0);
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("e)__A1dt0g+mp<@$!C", "bp9fqituD>-5a");
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0, 117);
      DataInputStream dataInputStream0 = new DataInputStream(pipedInputStream0);
      HttpConnection.KeyVal httpConnection_KeyVal1 = httpConnection_KeyVal0.inputStream((InputStream) dataInputStream0);
      httpConnection_Request1.data((Connection.KeyVal) httpConnection_KeyVal1);
      httpConnection_Request0.url = uRL0;
      try { 
        HttpConnection.Response.execute(httpConnection_Request1);
        fail("Expecting exception: ProtocolException");
      
      } catch(ProtocolException e) {
         //
         // Invalid HTTP method: PATCH
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      httpConnection_Request0.cookie("Content-Type", "\r\n\r\n");
      HttpConnection.Request httpConnection_Request1 = (HttpConnection.Request)httpConnection_Request0.cookie("\r\n\r\n", "\r\n\r\n");
      URL uRL0 = MockURL.getHttpExample();
      httpConnection_Request0.url = uRL0;
      try { 
        HttpConnection.Response.execute(httpConnection_Request1);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // HTTP error fetching URL
         //
         verifyException("org.jsoup.helper.HttpConnection$Response", e);
      }
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      HttpConnection.Request httpConnection_Request0 = new HttpConnection.Request();
      URL uRL0 = MockURL.getHttpExample();
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("\r\n\r\n", "\r\n\r\n");
      httpConnection_Request0.data((Connection.KeyVal) httpConnection_KeyVal0);
      httpConnection_Request0.data((Connection.KeyVal) httpConnection_KeyVal0);
      httpConnection_Request0.url = uRL0;
      try { 
        HttpConnection.Response.execute(httpConnection_Request0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // HTTP error fetching URL
         //
         verifyException("org.jsoup.helper.HttpConnection$Response", e);
      }
  }
}
